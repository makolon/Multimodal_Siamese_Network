docker run  -it --rm --gpus 1 --network host --privileged -e DISPLAY=$DISPLAY \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       -v /home/makolon/robo-trainer/:/home/robo-trainer/ \
       -v /dev/input/js0/:/dev/input/js0/ \
       --name robo-trainer robo_trainer

