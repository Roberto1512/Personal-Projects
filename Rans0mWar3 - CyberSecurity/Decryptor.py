from cryptography.fernet  import Fernet
import os
import argparse
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES, PKCS1_OAEP


class Decryptor:
    def __init__(self, key_path, root_path, pKey_path):
        self.key = self.read_fernet_key(key_path)
        self.crypter = Fernet(self.key)
        self.root_path = root_path
        self.pKey_path = pKey_path
        self.crypted_files = self.find_encrypted_files()

    def read_fernet_key(self, key_path):
        try:
            with open(key_path, 'rb') as f:
                enc_fernet_key = f.read()
            private_key = RSA.import_key(open(self.pKey_path).read())
            private_crypter = PKCS1_OAEP.new(private_key)
            fernet_key = private_crypter.decrypt(enc_fernet_key)
            return fernet_key
        except Exception as e:
            print(f"ERROR during fernet_key_file reading: {e}")

    def find_encrypted_files(self):
        encrypted_files =[]
        for root, _, files in os.walk(self.root_path, topdown=True):
            for file in files:
                if file.endswith('.encrypt'):
                    encrypted_files.append(os.path.join(root, file))
        return encrypted_files

    def decrypt_files(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
                _data = self.crypter.decrypt(data)
            with open(file_path, 'wb') as f:
                f.write(_data)
            new_path = file_path.rstrip('.encrypt')
            os.rename(file_path, new_path)
            print(f"{new_path} DECRIPTED")
        except Exception as e:
            print(f"ERROR with {new_path} file decription")
    
    def decrypt_system(self):
        for file in self.crypted_files:
            self.decrypt_files(file)
    
    def main():
        parser = argparse.ArgumentParser("!!!!!!!!!LAZARUS DECRYPTOR!!!!!!!!!")
        parser.add_argument("-k", "--key", type=str, required=True, help="decrypted fernet key file path")
        parser.add_argument("-d", "--directory", type=str, required=True, help="encrypted files path")
        parser.add_argument("-p", "--private", type=str, required=True, help="private key file path")
        args = parser.parse_args()
    
        if not os.path.isdir(args.directory):
            print(f"Directory not found")
            return
        
        decryptor = Decryptor(args.key, args.directory, args.private)
        decryptor.decrypt_system()

    if __name__ == "__main__":
        main()