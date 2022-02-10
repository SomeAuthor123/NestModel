
import subprocess
import pathlib



hep_link = r"http://snap.stanford.edu/data/cit-HepPh.txt.gz"
astro_link = r"http://snap.stanford.edu/data/ca-AstroPh.txt.gz"
# http://networksciencebook.com/translations/en/resources/data.html
networksciencebook_link = r"http://networksciencebook.com/translations/en/resources/networks.zip"
google_link = r"https://snap.stanford.edu/data/web-Google.txt.gz"
pokec_link = r"https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz"

networkscience_files = ['collaboration.edgelist.txt', 'powergrid.edgelist.txt', 'actor.edgelist.txt', 'www.edgelist.txt', 'phonecalls.edgelist.txt', 'internet.edgelist.txt', 'metabolic.edgelist.txt', 'email.edgelist.txt', 'citation.edgelist.txt', 'protein.edgelist.txt']
print(pathlib.Path(__file__).parent.absolute())


parent = pathlib.Path(__file__).parent.absolute()
print(parent.name)
if str(parent.name) == "scripts":
    parent = parent.parent
if str(parent.name)!="datasets":
    parent = parent/"datasets"
assert parent.is_dir()
assert parent.exists()
links = [hep_link, astro_link, networksciencebook_link, google_link, pokec_link]
download_names = ["cit-HepPh.txt.gz", "ca-AstroPh.txt.gz", "networks.zip", "web-Google.txt.gz", "soc-pokec-relationships.txt.gz"]
final_files = [("cit-HepPh.txt",), ("ca-AstroPh.txt",), networkscience_files, ("web-Google.txt",), ("soc-pokec-relationships.txt",)]


if True:
    files = list(file.name for file in parent.iterdir())
    for link, download_name, finals in zip(links, download_names, final_files):
        if all(final in files for final in finals):
            continue

        if not download_name in files:
            # download file
            command = "wget " + link +" -P " + str(parent)
            print()
            print("<<< downloading " + download_name)
            subprocess.call(command, shell=True, cwd=str(parent))

        if download_name.endswith(".gz"):
            command = "gzip -d " + str(download_name)
            print("<<< extracting " + download_name)
            subprocess.call(command, shell=True , cwd=str(parent))

print()
print("done")
print()

#import unnet
#from unnet.GEXFConverter.py import *

#out_name = out_names[0]
#with open(out_names, 'r'):