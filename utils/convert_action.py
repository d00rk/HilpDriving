import os, glob
import h5py

def convert_action(data_paths):
    for data in data_paths:
        with h5py.File(data, 'r+') as f:
            step_keys = sorted(list(f.keys()))
            for step_key in step_keys:
                action = f[step_key]['supervision']['action'][:]  # 기존 데이터 읽기
                throttle, steer, brake = action[:]
                acc = throttle - brake
                new_action = [acc, steer]

                # 기존 데이터셋 삭제
                del f[step_key]['supervision']['action']

                # 새 데이터셋 생성
                f[step_key]['supervision'].create_dataset('action', data=new_action)
                
                print(f"converted action in data: {data}")
            
if __name__ == "__main__":
    data_paths = list(glob.glob(os.path.join(os.getcwd(), "data/town*.hdf5")))
    convert_action(data_paths)
