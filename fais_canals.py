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

parser.add_argument("-c", "--no_center_scale", action = 'store_true', 
	            help = "Do not center and scale base weights at each locus.")

parser.add_argument("-n", "--num_cores", type = int, metavar = "int cores", default=0, 
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


def parseread(read_str, qual_str, ref_base):
 
	def base_prob(ascii, offset = 35):
		#ascii = ascii.replace('"','\"').replace("'","\'")
		return 1 - 1/np.power(10, (ord(ascii) - offset)/10)
    
	goods = ["A", "T", "G", "C", "N", "a", "t", "g", "c", "n"]
	nuc_dict = {"A": 0, "T": 0, "G": 0, "C":0, "N":0}
	i = 0
	q = 0
	while i < len(qual_str):
		base = read_str[i].replace(",", ref_base).replace(".", ref_base).upper()
		prob = base_prob(qual_str[q])
		if base == "^" :
			i += 1
		if base in ["+", "-"]:
			count = int(read_str[i+1])
			i += count + 1
		if base in goods:
			nuc_dict[base] += prob
			q += 1
		i += 1

	return list(nuc_dict.values())[0:4]

def filter_sites(site_list):
	idxs = list(range(0, len(site_list), 3))
	pop_str = ""
	dp_list = list()
	for i in idxs:
		pop_str += site_list[i+1]
		dp_list.append(int(site_list[i]))
	ref_prop = (pop_str.count(".") + pop_str.count(",") + pop_str.count("*")) / len(pop_str)
	dp_var = np.array(dp_list).std()
	return {"ref_prop": ref_prop, "dp_var": dp_var}


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
	cmd = "samtools mpileup -f {0} -l {1} -b bams.txt".format(args.reference, args.positions_file).split()
else:

	cmd = "samtools mpileup -f {0} -b bams.txt".format(args.reference).split()


proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
	line_list = line.strip().split("\t")
	ref = line_list[2]
	pos = '\t'.join(line_list[0:2])
	if ref in ["A", "T", "G", "C", "a", "t", "g", "c"]:
		ind_list = line_list[3:]
		qc = filter_sites(ind_list)
		if (qc["ref_prop"] < args.prop_variant or ( 1 - qc["ref_prop"]) > args.prop_variant) or qc["dp_var"] > args.depth_variance:
			quad_channel = list()
			inputs = list(range(1, len(line_list)-3, 3))
			result = Parallel(n_jobs = args.num_cores)(delayed(parseread)(ind_list[i], ind_list[i+1], ref) for i in inputs)
			quad_channel.append(np.asarray(result))
			quad_channel = np.squeeze(np.asarray(quad_channel))
			if args.no_center_scale == True:
				pass
			else:
				qsd = quad_channel.std()
				if qsd == 0: qsd = 1
				qmn =  quad_channel.mean()
				quad_channel = (quad_channel - qmn) / qsd

			for idx, arr in enumerate(quad_channel):
				f = tables.open_file(geno_pheno_df["outs"][idx], mode='a')
				f.root.data.append(arr.reshape(1, 4))
				f.close()
	
			if args.sites_file:
				pos_info = "{0}\t{1}\t{2}\n".format(pos, str(qc["ref_prop"]), str(qc["dp_var"]))
				args.sites_file.write(pos_info)
				args.sites_file.flush()

			else:
				print(pos)
			
