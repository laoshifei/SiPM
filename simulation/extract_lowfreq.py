"""Use the previously verified full-pool extractor in the new study directory."""
import argparse
import concurrent.futures
import extract_spectral_split as inherited
from lowfreq_common import *


def one(key):
    # Only the destination study directory changes. ADC, QC, charge and all
    # feature calculations are the frozen, previously verified implementation.
    inherited.OUT=OUT
    inherited.one(key)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    p=read(OUT/'protocol.json');code='simulation/extract_spectral_split.py'
    freeze=OUT/'extraction_protocol.json'
    if not freeze.exists():save(freeze,{'adapter_sha256':sha(__file__),'extractor_sha256':sha(PROJECT/code),'native_protocol_sha256':sha(OUT/'native_protocol.json')})
    fp=read(freeze);assert fp['adapter_sha256']==sha(__file__) and fp['extractor_sha256']==sha(PROJECT/code)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(one,list(p['entries'])):pass
