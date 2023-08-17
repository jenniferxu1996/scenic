import pandas as pd
import os
import cv2
import json

def generate_split_cooking():
    CLIP_PATH = 'cooking_video_clip'
    subject_folders = os.listdir(CLIP_PATH)
    train_annotation_file = pd.DataFrame(columns=['video_path', 'start', 'end', 'label'])
    valid_annotation_file = pd.DataFrame(columns=['video_path', 'start', 'end', 'label'])
    test_annotation_file = pd.DataFrame(columns=['video_path', 'start', 'end', 'label'])
    all_annotation_file = pd.DataFrame(columns=['video_path', 'start', 'end', 'label'])

    # train_subject = ['s07-d72-cam-002', 's08-d02-cam-002', 's13-d28-cam-002']
    # valid_subject = ['s14-d61-cam-002']
    # test_subject = ['s26-d23-cam-002']
    valid_subject = ['s21-d21-cam-002', 's21-d23-cam-002', 's21-d27-cam-002', 's21-d28-cam-002', 's21-d29-cam-002', 's21-d35-cam-002',
                     's21-d39-cam-002', 's21-d40-cam-002', 's21-d42-cam-002', 's21-d43-cam-002', 's21-d45-cam-002', 's21-d49-cam-002',
                     's21-d50-cam-002', 's21-d52-cam-002', 's21-d53-cam-002', 's21-d55-cam-002', 's21-d63-cam-002']
    test_subject = ['s22-d23-cam-002', 's22-d25-cam-002', 's22-d26-cam-002', 's22-d29-cam-002', 's22-d31-cam-002', 's22-d34-cam-002',
                    's22-d35-cam-002', 's22-d43-cam-002', 's22-d46-cam-002', 's22-d48-cam-002', 's22-d53-cam-002', 's22-d55-cam-002',
                    's28-d23-cam-002', 's28-d25-cam-002', 's28-d27-cam-002', 's28-d39-cam-002', 's28-d46-cam-002', 's28-d51-cam-002',
                    's28-d70-cam-002', 's28-d74-cam-002', 's29-d29-cam-002', 's29-d31-cam-002', 's29-d39-cam-002', 's29-d42-cam-002',
                    's29-d49-cam-002', 's29-d50-cam-002', 's29-d52-cam-002', 's29-d71-cam-002', 's33-d23-cam-002', 's33-d27-cam-002',
                    's33-d45-cam-002', 's33-d49-cam-002', 's33-d50-cam-002', 's33-d54-cam-002', 's34-d21-cam-002', 's34-d28-cam-002',
                    's34-d34-cam-002', 's34-d40-cam-002', 's34-d41-cam-002', 's34-d63-cam-002', 's34-d69-cam-002', 's34-d73-cam-002']

    for subject_folder in subject_folders:
        activity_folder = os.path.join(CLIP_PATH, subject_folder)
        activity_videos = os.listdir(activity_folder)
        for activity_video in activity_videos:
            activity_video_path = os.path.join(activity_folder, activity_video)

            cap = cv2.VideoCapture(activity_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            cap.release()

            print(subject_folder, activity_video, flush=True)

            label = os.path.splitext(activity_video)[0].split("_")[0]
            new_row = pd.DataFrame({'video_path': [activity_video_path], 'start': [0.0], 'end': [duration], 'label': [label]})
            if subject_folder in valid_subject:
                valid_annotation_file = pd.concat([valid_annotation_file, new_row], ignore_index=True)
            elif subject_folder in test_subject:
                test_annotation_file = pd.concat([test_annotation_file, new_row], ignore_index=True)
            else:
                train_annotation_file = pd.concat([train_annotation_file, new_row], ignore_index=True)
            all_annotation_file = pd.concat([all_annotation_file, new_row], ignore_index=True)

    train_annotation_file.to_csv('cooking_annotation_train.csv', index=False)
    valid_annotation_file.to_csv('cooking_annotation_valid.csv', index=False)
    test_annotation_file.to_csv('cooking_annotation_test.csv', index=False)
    all_annotation_file.to_csv('cooking_annotation.csv', index=False)


def generate_split_something():
    CLIP_PATH = 'something_video_clip'
    videos = os.listdir(CLIP_PATH)
    train_annotation_file = pd.DataFrame(columns=['video_path', 'start', 'end', 'label'])
    valid_annotation_file = pd.DataFrame(columns=['video_path', 'start', 'end', 'label'])
    test_annotation_file = pd.DataFrame(columns=['video_path', 'start', 'end', 'label'])
    all_annotation_file = pd.DataFrame(columns=['video_path', 'start', 'end', 'label'])

    train_label = pd.read_json('something_labels/train.json')
    train_id = train_label['id'].tolist()
    test_label = pd.read_csv('something_labels/test-answers.csv', sep=';')
    test_id = test_label['id'].tolist()
    valid_label = pd.read_json('something_labels/validation.json')
    valid_id = valid_label['id'].tolist()

    for video in videos:
        video_path = os.path.join(CLIP_PATH, video)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()
        video_id = int(os.path.splitext(video)[0])
        if video_id in valid_id:
            video_info_row = valid_label[valid_label['id'] == video_id]
            mode = 'valid'
        elif video_id in test_id:
            video_info_row = test_label[test_label['id'] == video_id]
            mode = 'test'
        else:
            video_info_row = train_label[train_label['id'] == video_id]
            mode = 'train'
        if 'template' in video_info_row.columns:
            activity_label = video_info_row['template'].tolist()[0].replace("[", "").replace("]", "")
            new_row = pd.DataFrame({'video_path': [video_path], 'start': [0.0], 'end': [duration], 'label': [activity_label]})
            if mode == 'valid':
                valid_annotation_file = pd.concat([valid_annotation_file, new_row], ignore_index=True)
            elif mode == 'test':
                test_annotation_file = pd.concat([test_annotation_file, new_row], ignore_index=True)
            else:
                train_annotation_file = pd.concat([train_annotation_file, new_row], ignore_index=True)
    train_annotation_file.to_csv('something_annotation_train.csv', index=False)
    valid_annotation_file.to_csv('something_annotation_valid.csv', index=False)
    test_annotation_file.to_csv('something_annotation_test.csv', index=False)


def generate_label(dataset):
    assert dataset in ['cooking', 'something']
    if dataset == 'cooking':
        all_annotation_file = pd.read_csv('cooking_annotation.csv')
        labels = all_annotation_file['label'].unique().tolist()
        with open('label.txt', 'w') as f:
            for label in labels:
                f.writelines(label)
                f.write('\n')
        f.close()
    elif dataset == 'something':
        with open('something_labels/labels.json', 'r') as json_file:
            data = json.load(json_file)
        with open('something_labels/label.txt', 'w') as f:
            for key in data.keys():
                f.writelines(key)
                f.write('\n')
        f.close()


if __name__ == '__main__':
    generate_label('something')
