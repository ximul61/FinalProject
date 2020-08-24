import os
import librosa
import json

dataset_path = "Dataset"
sample_rate = 22050
JSON_PATH = "data_10.json"


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512):
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }
    num = 0
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print(f"\n\n\nProcessing:  {semantic_label}")

            for f in filenames:
                num = num + 1
                file_path = os.path.join(dirpath, f)
                # Loading the dataset
                signal, sr = librosa.load(file_path, sr=sample_rate)

                # Extracting MFCCs for future work
                mfcc = librosa.feature.mfcc(signal,
                                            sr=sr,
                                            n_fft=n_fft,
                                            n_mfcc=n_mfcc,
                                            hop_length=hop_length)
                mfcc = mfcc.T
                data["mfcc"].append(mfcc.tolist())
                data["labels"].append(i - 1)

                print(f'Segments: {file_path} --------- {num}')

    # Saving The MFCCs in a json file to train our model.
    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(dataset_path, JSON_PATH)


# https://zenodo.org/record/1290750/files/IRMAS-TrainingData.zip?download=1