"""
This is a utility script used to play in the OHT with a trained 7x7 Hex model.

Usage: python oht.py -m <path/to/7x7/hex/model>
"""

import argparse

from oht_client.hex_client import HexClient

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='path/to/7x7/model', type=str)
parser.add_argument('-l', '--league', action='store_true')
args = parser.parse_args()

client = HexClient(model_path=args.model)

if args.league:
    client.run(mode='league')
else:
    client.run()