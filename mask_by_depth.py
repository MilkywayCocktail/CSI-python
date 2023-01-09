import numpy as np
import tqdm


def mask_by_depth(data):
    median = np.median(data, axis=0)
    threshold = median * 0.5

    out = np.zeros_like(data)

    for i in tqdm.tqdm(range(len(data))):
        mask = data[i] < threshold
        masked = data[i] * mask
        out[i] = masked

    return out


def run(in_path, out_path):
    data = np.load(in_path)
    print(data.shape)

    median = np.median(data, axis=0)
    threshold = median * 0.5

    out = np.zeros_like(data)

    for i in tqdm.tqdm(range(len(data))):
        mask = data[i] < threshold
        masked = data[i] * mask
        out[i] = masked

    print(out[0])
    np.save(out_path, out)
    print("Saved!")


if __name__ == '__main__':
    in_path = '../dataset/compressed/121304.npy'
    out_path = '../dataset/compressed/121304_masked.npy'
    run(in_path, out_path)
    #print(np.load(out_path).shape)
