import argparse
import numpy as np
import os
import re
import io
import subprocess
import tables
import pandas as pd

from joblib import Parallel, delayed
import multiprocessing

#from parse_read import parse_read

parser = argparse.ArgumentParser(description="Produce quad channel representation from sorted bams")

parser.add_argument("-p", "--prop_variant", type = float, metavar = "prop", default=1, 
		    help = "Floating point value between 0.5 and 1 to exclude sites with little or no sequence variation (less than prop).")

parser.add_argument("-d", "--depth_variance", type = float, metavar = "dp", default=0.0, 
	            help = "Positive real value to exclude sites with little variation in base counts (less than dp).")

parser.add_argument("-s", "--distance_between_sites", type = int, metavar = "space", default=1,
                    help = "Positive integer specifying the minimum distance between variant sites to be included.")

parser.add_argument("-c", "--no_center_scale", action = 'store_true', 
	            help = "Do not center and scale base weights at each locus.")

parser.add_argument("-n", "--num_cores", type = int, metavar = "int cores", default=1, 
		    help = "Number of cores to use. If multiple cores are not available, do not use flag, or set int cores to zero")

parser.add_argument('-o', '--sites_file', nargs="?", type=argparse.FileType('w'), 
		    help='Optional name of the file to write included loci.')

parser.add_argument('-l', '--positions_file', type=str, 
		    help='Optional file of reference position to pass to samtools mpileup.')

parser.add_argument("reference", type=str, help="The indexed reference genome each bam was aligned to")

parser.add_argument("pheno_geno_map", type=str, help="Data frame (where column names MUST match those following in parentheses) matching paths to bam files (bams), phenotypes (phenos), our desired H5 output file names (outs)")

args = parser.parse_args()


geno_pheno_df = pd.read_csv(args.pheno_geno_map, sep = "\t")
geno_pheno_df["bams"].to_csv("bams.txt", sep = "\t", header = False, index = False)


def parse_line(pileup):
	mp = pileup.strip().split("\t")
	chrom, pos, ref = mp[0:3] #site data
	pop_bam = mp[3:]
	idx = list(range(0,len(pop_bam), 3))
	site_dict = {"chrom": chrom, "ref": ref, "pos": pos, "pop_bam": pop_bam, "idx":idx}
	return site_dict

def raw_dp(pop_bam, idx):
	#set up indices in groups of 3
	#process read strings
	dps =  np.array([int(pop_bam[i]) for i in idx])
	dp_var =  dps.var()
	return {"dp_var": dp_var, "dps": dps}

def raw_freq(pop_bam, dps, idx):
	read_list = [pop_bam[i+1] for i in idx]
	read_str = ' '.join(read_list).replace("*", "")
	#print(read_str)
	ref_freq = (read_str.count(",") + read_str.count(".")) / np.sum(dps)
	return ref_freq
	#freq_test = (1 - args.ref_freq) <= ref_freq <= args.ref_freq
        


def base_prob(phred, offset = 35):
	return np.array([1 - 1/np.power(10, (ord(asci) - offset)/10) for asci in phred])

def parse_seq(seq_str, qual_str, ref):
    
	seq_str = re.sub('\^.', '', seq_str.replace(".", ref).replace(",", ref))
	seq_str = seq_str.replace("$", "")
    
	broken_seq = re.split('[+-]\d+', seq_str)
    
	seq_array =  np.array(list(broken_seq[0] + ''.join(
	[
	broken_seq[i][int(re.findall('\d+', seq_str)[i-1]):]
		for i in list(range(1, len(broken_seq)))
	]
	)
	))

	qs = base_prob(qual_str, offset = 35)
	return [
		qs[seq_array == "A"].sum(),
		qs[seq_array == "T"].sum(),
		qs[seq_array == "G"].sum(),
		qs[seq_array == "C"].sum()
		]


def multiple_replace(dict_rep, text):
	# Create a regular expression  from the dictionary keys
	regex = re.compile("(%s)" % "|".join(map(re.escape, dict_rep.keys())))
	# For each match, look-up corresponding value in dictionary
	return regex.sub(lambda mo: dict_rep[mo.string[mo.start():mo.end()]], text) 


replacers = dict = {
	"sorted" : "",
	"bam" : "",
	"." : "",
	"/" : ""} 


#generate H5 files to append to as mpileup is read from stdin
for outs in geno_pheno_df["outs"]:
	f = tables.open_file(outs, mode='w')
	atom = tables.Float64Atom()
	array_c = f.create_earray(f.root, 'data', atom, (0, 4))
	f.close()

#!!!!!!
if args.positions_file:
	cmd = "samtools mpileup -A -f {0} -l {1} -b bams.txt".format(args.reference, args.positions_file).split()
else:
	cmd = "samtools mpileup -A -f {0} -b bams.txt".format(args.reference).split()


space_count = 1
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
	if space_count >= args.distance_between_sites:
		canal = list()
		site_dict = parse_line(line)
		chrom = site_dict["chrom"]
		pos = site_dict["pos"]
		pop_bam = site_dict["pop_bam"]
		idx = site_dict["idx"]
		ref = site_dict["ref"]	
		dp_res = raw_dp(pop_bam, idx)
		#print(dp_res)
		if dp_res["dp_var"] >= 0:
			freq_res = raw_freq(pop_bam, dp_res["dps"], idx)
			#print(line.strip())
			if (1-args.prop_variant) <= freq_res <= args.prop_variant or dp_res["dp_var"] >= args.depth_variance:
				#print(line.strip())
				res = Parallel(n_jobs = args.num_cores)(delayed(parse_seq)(pop_bam[i+1], pop_bam[i+2], ref) for i in idx)
				canal.append(res)
				canal = np.squeeze(np.asarray(canal))
				if args.no_center_scale == True:
					pass
				else:
					qsd = canal.std()
					qmn = canal.mean()
					canal = ((canal - qmn)/qsd).round(4)
				for idx, arr in enumerate(canal):
					f = tables.open_file(geno_pheno_df["outs"][idx], mode='a')
					f.root.data.append(arr.reshape(1, 4))
					f.close()
				
				pos_info = "{0}\t{1}\t{2}\t{3}\n".format(chrom, pos, str(freq_res), str(dp_res["dp_var"]))
				if args.sites_file:
					args.sites_file.write(pos_info)
					args.sites_file.flush()
				else:
					print(pos_info)
				space_count = 1
	else:
		space_count+=1
				
	
		#check if propotion of reference alleles is between specified user cutoff OR variance in depth in above user cutoff
#		if space_count >= args.distance_between_sites and ((1 -  args.prop_variant) < qc["ref_prop"] < args.prop_variant or qc["dp_var"] > args.depth_variance):
#			quad_channel = list()
#			inputs = list(range(1, len(line_list)-3, 3))
#			result = Parallel(n_jobs = args.num_cores)(delayed(parseread)(ind_list[i], ind_list[i+1], ref) for i in inputs)
#			quad_channel.append(np.asarray(result))
#			quad_channel = np.squeeze(np.asarray(quad_channel))
#			if args.no_center_scale == True:
#				pass
#			else:
#				qsd = quad_channel.std()
#				if qsd == 0: qsd = 1
#				qmn =  quad_channel.mean()
#				quad_channel = (quad_channel - qmn) / qsd
#
#			for idx, arr in enumerate(quad_channel):
#				f = tables.open_file(geno_pheno_df["outs"][idx], mode='a')
#				f.root.data.append(arr.reshape(1, 4))
#				f.close()
	
#			if args.sites_file:
#				pos_info = "{0}\t{1}\t{2}\n".format(pos, str(qc["ref_prop"]), str(qc["dp_var"]))
#				args.sites_file.write(pos_info)
#				args.sites_file.flush()
#
#			else:
#				print(pos)
#			space_count = 1
#		else:
#			space_count += 1




#def parseread(read_str, qual_str, ref_base):
#
#	def base_prob(ascii, offset = 35):
#		#ascii = ascii.replace('"','\"').replace("'","\'")
#		return 1 - 1/np.power(10, (ord(ascii) - offset)/10)
#   
#	goods = ["A", "T", "G", "C", "N", "a", "t", "g", "c", "n"]
#	nuc_dict = {"A": 0, "T": 0, "G": 0, "C":0, "N":0}
#	i = 0
#	q = 0
#	while i < len(qual_str):
#		base = read_str[i].replace(",", ref_base).replace(".", ref_base).upper()
#		prob = base_prob(qual_str[q])
#		if base == "^" :
#			i += 1
#		if base in ["+", "-"]:
#			count = int(read_str[i+1])
#			i += count + 1
#		if base in goods:
#			nuc_dict[base] += prob
#			q += 1
#		i += 1
#
#	return list(nuc_dict.values())[0:4]

#def filter_sites(site_list):
#	idxs = list(range(0, len(site_list), 3))
#	pop_str = ""
#	dp_list = list()
#	for i in idxs:
#		pop_str += site_list[i+1]
#		dp_list.append(int(site_list[i]))
#	ref_prop = (pop_str.count(".") + pop_str.count(",") + pop_str.count("*")) / len(pop_str)
#	dp_var = np.array(dp_list).std()
#	return {"ref_prop": ref_prop, "dp_var": dp_var}


