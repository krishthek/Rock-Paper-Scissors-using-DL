
import numpy as np
import cv2 
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import h5py
from keras.models import model_from_json
import json
import random

label_to_class = {0:'rock,', 1:'paper', 2:'scissor'}

def choose_winner(player1, player2):
    '''Given player1 and player2, choose who wins the game of rock,paper, scissor'''
    if player1==player2:
        return 'tie'

    if player1 == 'rock':
        if player2 == 'paper':
            return 'player2'
        else:
            return 'player1'

    if player1 == 'paper':
        if player2 == 'scissor':
            return 'player2'
        else:
            return 'player1'
    
    if player1 == 'scissor':
        if player2 == 'rock':
            return 'player2'
        else:
            return 'player1'
    




def fix_layer0(filename, batch_input_shape, dtype):
    with h5py.File(filename, 'r+') as f:
        model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
        layer0 = model_config['config']['layers'][0]['config']
        layer0['batch_input_shape'] = batch_input_shape
        layer0['dtype'] = dtype
        f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')

fix_layer0('rps_vgg16.h5', [None, 224, 224, 3], 'float32')

loaded_model = load_model('rps_vgg16.h5')

print("Loaded model from disk")
#loaded_model.summary()

#the image size for input is (224,224)
img_size = 224

player_score = 0
computer_score = 0


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if return value is false, we skip the block
    if not ret:
        continue
    
    
    # insert rectangle box
    cv2.rectangle(frame,(60,60),(300,300),(255,0,0),2)

    #setting up our region of interest from where we want the pictures to be taken
    roi = frame[60:300, 60:300]
    #flip the image
    frame = cv2.flip(frame,1)

    # Display the resulting frame
    #cv2.imshow('frame',frame)

    # do the preprocessing requires
    img_arr = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img_arr = cv2.resize(img_arr, (img_size,img_size))
    img_arr = np.array(img_arr).reshape(-1, img_size,img_size,3)
    img_arr = preprocess_input(img_arr)

    #predict the play
    y_prob = loaded_model.predict(img_arr) 
    y_label = np.argmax(y_prob)
    player_move = label_to_class[y_label]

    computer_move = random.choice(['rock','paper', 'scissor'])

    #added things
    # if space is clicked the a game is played
    if cv2.waitKey(1) & 0xFF==ord(' '):
        cv2.putText(frame, "Compter move: {}".format(computer_move),(300, 50), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
        winner = choose_winner(player_move, computer_move)
        if winner == 'player1':
            player_score +=1
        else:
            computer_score +=1
    
    #code to output and show the details 
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Player move: {}".format(player_move),(5, 50), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
    #cv2.putText(frame, "Compter move: {}".format(computer_move),(300, 50), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Player Score: {}".format(player_score),(120, 400), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Computer Score: {}".format(player_score),(300, 400), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("move is", frame)
    
        
        
    # press esc to end the capture
    k =  cv2.waitKey(1) & 0xFF 
    if k==27:
        break
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()