import pickle
import os

def inspect_pickle_structure(file_path):
    print(f"=== Inspecting structure of {file_path} ===")
    if not os.path.exists(file_path):
        print("File not found.")
        return

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Type of data: {type(data)}")
    if isinstance(data, list):
        print(f"Length of list: {len(data)}")
        if len(data) > 0:
            print(f"Type of first element: {type(data[0])}")
            print(f"First element content: {data[0]}")
    elif isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
    else:
        print(f"Content: {data}")

if __name__ == "__main__":
    inspect_pickle_structure("data/processed_ck/receiver_train/data.pickle")

