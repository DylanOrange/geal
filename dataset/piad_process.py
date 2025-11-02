import os
import numpy as np
import pickle

AFFORDANCE = ['grasp', 'contain', 'lift', 'open', 
                        'lay', 'sit', 'support', 'wrapgrasp', 'pour', 'move', 'display',
                        'push', 'listen', 'wear', 'press', 'cut', 'stab']


def extract_point_file(path):
    with open(path,'r') as f:
        coordinates = []
        lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        line = line.strip(' ')
        data = line.split(' ')
        coordinate = [float(x) for x in data[2:]]
        coordinates.append(coordinate)
    data_array = np.array(coordinates)
    points_coordinates = data_array[:, 0:3]
    affordance_label = data_array[: , 3:]

    return points_coordinates, affordance_label

def main(args):

    train_split = f'{args.setting}/Point_{args.split}.txt'
    file_list = []
    with open(os.path.join(args.dataroot, train_split),'r') as f:
        files = f.readlines()
        for file in files:
            file = file.strip('\n')
            file_list.append(file)
        f.close()

    file_list = [path.replace('Data', args.dataroot, 1) for path in file_list]

    data_list = []
    sample_num = 0
    for point_file in file_list:
        class_label = point_file.split('/')[-2]

        Points, all_mask = extract_point_file(point_file)
        for idx in range(all_mask.shape[1]):
            if np.sum(all_mask[:,idx]) > 0:
                affordance_label = AFFORDANCE[idx]
                gt_mask = all_mask[:,idx]
                if affordance_label == 'wrapgrasp':
                    affordance_label = 'wrap_grasp'
                data = {
                    'class': class_label.lower(),
                    'affordance': affordance_label,
                    'point': Points.astype(np.float32),
                    'mask': gt_mask.astype(np.float32),
                }
                data_list.append(data)
                sample_num += 1

    print('Total sample number:', sample_num)
    
    setting_name = args.setting.lower()
    split_name = args.split.lower()
    output_path = os.path.join(args.dataroot, f'{setting_name}_{split_name}.pkl')

    with open(output_path, 'wb') as f:
        pickle.dump(data_list, f)

    print(f"Saved dataset to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="piad_dataset")
    parser.add_argument("--setting", type=str, default="Unseen", help="Seen or Unseen")
    parser.add_argument("--split", type=str, default="Test", help="Train or Test")
    args = parser.parse_args()
    main(args)

