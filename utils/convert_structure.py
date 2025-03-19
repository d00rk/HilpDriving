import h5py
import os
import glob
DATA_PATH = glob.glob(os.path.join(os.path.dirname(os.getcwd()), 'datasets/roach_bc/leaderboard1/*.hdf5'))

def main():
    for dataset in DATA_PATH:
        with h5py.File(dataset, 'r+') as f:
            original_keys = list(f.keys())
            # print(f"original keys from {dataset}: {original_keys}")
            
            temp_mapping = {}
            for key in original_keys:
                temp_key = "temp_" + key
                f.move(key, temp_key)
                temp_mapping[temp_key] = key
                
            for temp_key, original_key in temp_mapping.items():
                number = int(original_key.split('_')[-1])
                new_key = "step_{:04d}".format(number)
                f.move(temp_key, new_key)
                
            print(f"converted keys from {dataset}: {list(f.keys())}")
            
if __name__ == "__main__":
    main()