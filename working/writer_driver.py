import punk_rock_writer as prw
import os


def main(seed='I '):

    data_files = []
    while True:
        data_file = input('Please input a date file: ')
        if data_file == 'done':
            break
        data_files.append(data_file)

    epochs = []
    while True:
        try:
            epoch_count = int(input('Please add a number of epochs you would like to train: '))
            if epoch_count <= 0:
                break
            epochs.append(epoch_count)
        except ValueError:
            print('That was not a valid integer')

    for data_file in data_files:

        data_dir = './data/' + data_file
        base_dir = os.getcwd() + '/models_with_output/working/' + data_file
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)

        for epoch in epochs:

            epoch_dir = f'{base_dir}/{epoch}epoch(s)'
            if not os.path.exists(epoch_dir):
                os.mkdir(epoch_dir)

            model_num = len(os.listdir(epoch_dir))
            model_dir = f'{epoch_dir}/model{model_num}'
            os.mkdir(model_dir)
            output_dir = model_dir + '/output'
            os.mkdir(output_dir)

            model_name = f'{data_file}_{epoch}epoch(s)_model{model_num}'
            prw.main(data_dir, epoch, model_name, seed,
                     model_dir, output_dir + '/' + model_name + '_output1')


if __name__ == '__main__':
    main()
