import sentencepiece as spm
import argparse

def main():
  parser = argparse.ArgumentParser(description="Detokenizer")

  # Add arguments
  parser.add_argument("token_id", type=int, help="token id")

  # Parse the arguments
  args = parser.parse_args()

  # Access the argument values
  token_id = args.token_id

  print(f"Detokenizing {token_id}...")

  sp = spm.SentencePieceProcessor()
  sp.load("/home/sa_112155357684894056033/maxtext/assets/tokenizer.mistral-v3")

  decode_token_list = [token_id]
  print("Decoded token: ", sp.decode_ids(decode_token_list))
  
if __name__ == "__main__":
  main()


