##Readme

Download 'HandAction.tar', 'Train1.tar', and 'Train2.tar'. 
Firstly please extract 'HandAction.tar' into a folder **$data_path$**, and then extract the contents of 'Train1.tar' and 'Train2.tar' into '**$data_path$**/Train/' folder.

In 'Train/' and 'Test/' folders, there exist a list of subfolders, and each subfolder corresponds a subject. 

In each subject folder there exist 16 sequences which correspond to 16 actions.

Each sequence contains two parts:

* $seq_name$.xml      (label file)

* $seq_name$/         (a folder contains images)

In each $seq_name$ folder, there exist 3 series of images:

* depth_#.png     (depth image (16 bit 1 channel))
    
* confi_#.png     (IR image (16 bit 1 channel))
    
* color_#.png     (rgb image (8 bit 3 channel))
    
where # is zero base frame index

In 'TestDetection' foler, there exist 10 sequences.

Details of the dataset are as follows:

##Training data
Train/    

    - Number of Subjects : 15
    - Number of gestures : 16
    - Total number of frames : 112,640

    - Number of frames for each gesture:
        asl-bathroom : 6427
        asl-blue : 7349
        asl-green : 7919
        asl-j : 7604
        asl-milk : 7111
        asl-scissors : 6925
        asl-where : 7589
        asl-yellow : 7742
        asl-you : 7084
        asl-z : 8667
        ui-circle : 7876
        ui-click : 5547
        ui-doubleclick : 6128
        ui-keyTap : 5690
        ui-screenTap : 5926
        ui-swipe : 7056

    - Number of frames for each subject:
        LN : 7832
        aiyang : 6723
        ashwin : 8097
        biru : 8207
        chi : 9454
        chris : 8163
        gulin : 5274
        justin : 7351
        lakshmi : 6881
        malay : 6373
        marc : 6003
        michael : 6210
        xuchi : 11000
        yizhou : 9106
        yongzhong : 5966


##Testing data for recognition
Test/    

    - Number of Subjects : 3
    - Number of gestures : 16
    - Total number of frames : 22,025

    - Number of frames for each gesture:
        asl-bathroom : 1458
        asl-blue : 1411
        asl-green : 1319
        asl-j : 1500
        asl-milk : 1477
        asl-scissors : 1316
        asl-where : 1448
        asl-yellow : 1483
        asl-you : 1439
        asl-z : 1724
        ui-circle : 1464
        ui-click : 1107
        ui-doubleclick : 1153
        ui-keyTap : 1165
        ui-screenTap : 1200
        ui-swipe : 1361

    - Number of frames for each subject:
        alvin : 8239
        etienne : 6119
        xiaowei : 7667


##Testing data for detection and recognition
TestDetection/    

    - Number of Subjects : 9
    - Number of gestures : 16
    - Total number of frames : 19,537

    - Number of instances of each gesture:
        asl-you : 4
        ui-circle : 7
        ui-swipe : 15
        asl-j : 10
        ui-doubleclick : 9
        asl-bathroom : 6
        asl-where : 6
        asl-scissors : 11
        ui-click : 10
        asl-z : 15
        asl-green : 6
        asl-milk : 7
        ui-keyTap : 8
        asl-blue : 13
        ui-screenTap : 6
        asl-yellow : 6

    - Number of frames for each subject:
        alvin_test : 1064
        connie : 1894
        dale0 : 2386
        dale1 : 2455
        lam : 2313
        nguyen : 1634
        prashanth : 1432
        vishnu : 2503
        xiaowei_test : 1727
        zhaohe : 2129

