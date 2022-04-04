import argparse

from oht_client.hex_client import HexClient

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='path/to/7x7/model', type=str)
args = parser.parse_args()

client = HexClient(model_path=args.model)
client.run()