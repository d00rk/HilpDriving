import os, sys, glob
import h5py

data_paths = list(glob.glob("town*.hdf5"))

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
