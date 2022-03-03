from data.vggish_input import wavfile_to_examples


def generate_audioSamples(file_path, sample_path):

    with open(file_path) as f:
            lines = f.read().splitlines()
    audio_paths = []
    for line in lines:
        audio_filename, label = line.split(".mkv")
        audio_path = './dataset/' + audio_filename + '.wav'
        audio_paths.append((audio_path, int(label)))

    for path in audio_paths:
        audio_path, label = path
        print(audio_path)
        wavfile_to_examples(audio_path)




if __name__ == "__main__":

    DATA_PATH = "./dataset/set_splits/cmd3_labels.csv"
    SAMPLE_PATH = "./dataset/audio_samples_new"

    generate_audioSamples(DATA_PATH, SAMPLE_PATH)