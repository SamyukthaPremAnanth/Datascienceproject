from tkinter import *
import tkinter.font as tkFont
import tensorflow as tf
import cv2
import numpy as np
from tkinter import filedialog


def browseFiles():
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select a File",
                                          filetypes=(("Video files","*.mp4*"),("Video files","*.avi*")))
    model = tf.keras.models.load_model("E:/model_weight.hdf5")
    inception_v3 = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    x = inception_v3.output
    pooling_output = tf.keras.layers.GlobalAveragePooling2D()(x)
    feature_extraction_model = tf.keras.Model(inception_v3.input, pooling_output)

    def extraction(video_path, feature_extraction_model):
        print("    going to process " + str(video_path))

        features = []
        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_every_frame = max(1, num_frames // 40)
        max_images = 40

        for current_frame_index in range(num_frames):

            success, frame = cap.read()

            if not success:
                break

            if current_frame_index % sample_every_frame == 0:
                frame = frame[:, :, ::-1]

                img = tf.image.resize(frame, (299, 299))

                img = tf.keras.applications.inception_v3.preprocess_input(img)

                tensor_input = tf.expand_dims(img, axis=0)

                if False:
                    with tf.Session() as sess:
                        sess.run(init_op)

                cf = feature_extraction_model(tensor_input)
                cf = tf.reshape(cf, (cf.shape[0], -1))
                cf = cf.numpy()
                features.append(cf)

                max_images -= 1

            if max_images == 0:
                break

        return features

    video = filename
    features = extraction(video, feature_extraction_model)
    tensor_input = tf.convert_to_tensor(features, dtype=tf.float32)
    prediction = model.predict(tensor_input, verbose=1)
    i = 0
    for p in prediction:
        i += 1
    a = np.argmax(prediction[1])
    b = np.argmax(prediction[29])
    c = np.argmax(prediction[39])
    classes = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching',
               'BaseballPitch',
               'Basketball', 'BasketballDunk', 'BenchPress', 'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles',
               'BodyWeightSquats', 'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth',
               'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen', 'Diving', 'Drumming',
               'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', 'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut',
               'Hammering', 'HammerThrow', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump',
               'HorseRace',
               'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope',
               'Kayaking', 'Knitting', 'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', 'Nunchucks',
               'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute',
               'PlayingGuitar',
               'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps',
               'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin',
               'ShavingBeard',
               'Shotput', 'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty',
               'StillRings',
               'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus',
               'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups',
               'WritingOnBoard', 'YoYo']
    print(classes[b])
    print(classes[a])
    print(classes[c])

    label1.configure(text="Activity: " + classes[b])

window = Tk()
window.title('Dash Board')
window.geometry("1000x1000")
window.config(background="powder blue")
fontStyle = tkFont.Font(family="Helvetica",size=20,slant="italic",weight="bold")
fs=tkFont.Font(family="Helvetica",size=10,weight="bold")
label2 = Label(window,text="Human Activity Detection",height="3",fg="red",font=fontStyle)
label2.config(width=65)
button1 = Button(window,text="Choose the Video",command=browseFiles,font=fs).place(x=500,y=200)
button2= Button(window,text="Exit",command=exit,font=fs).place(x=540,y=250)
label2.grid(column=1, row=1)
label1 = Label(window,text=" ",fg="red",font=fontStyle)
label1.config(width=65)
label1.grid(column=1, row=3)
window.mainloop()