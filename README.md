# Gazing at Social Interactions Between Foraging and Decision Theory

***Alessandro D'Amelio¹, Giuseppe Boccignone¹***  
¹ [PHuSe Lab](https://phuselab.di.unimi.it) - Dipartimento di Informatica, Università degli Studi di Milano  

**Paper** D'Amelio, A., & Boccignone, G. (2021). [Gazing at social interactions between foraging and decision theory](https://www.frontiersin.org/articles/10.3389/fnbot.2021.639999/full?&utm_source=Email_to_authors_&utm_medium=Email&utm_content=T1_11.5e1_author&utm_campaign=Email_publication&field=&journalName=Frontiers_in_Neurorobotics&id=639999). Frontiers in Neurorobotics, 15, 31.

![simulation](simulation.gif "Model Simulation")

### Requirements

```
pip install -r requirements.txt
```

### Executing the demo

To simulate from the model:

1. Do speaker identification and build face maps:
```
#sh build_face_maps.sh path/to/video vidName path/to/output
sh build_face_maps.sh data/videos/012.mp4 012 speaker_detect/output/
```
2. Run the follwing command (it is assumed that low-level saliency maps (see Credits) are already computed, if you want to compute it on your own, you may want to use something like [this](https://users.soe.ucsc.edu/~milanfar/research/rokaf/.html/SaliencyDetection.html#Matlab))
```
python3 start_simulation.py
```

### Credits

- Data: FindWhoToLookAt --->[Repo](https://github.com/yufanLiu/find), [Paper](https://ieeexplore.ieee.org/document/8360155)

- Speaker Detection Pipeline: SyncNet ---> [Repo](https://github.com/joonson/syncnet_python), [Paper](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf)

- Space-time Visual Saliency Detection: [Code](https://users.soe.ucsc.edu/~milanfar/research/rokaf/.html/SaliencyDetection.html#Matlab), [Paper](http://jov.arvojournals.org/article.aspx?articleid=2122209)

### Reference

If you use this code, please cite the paper:
```
@article{d2021gazing,
  title={Gazing at social interactions between foraging and decision theory},
  author={D'Amelio, Alessandro and Boccignone, Giuseppe},
  journal={Frontiers in Neurorobotics},
  volume={15},
  pages={31},
  year={2021},
  publisher={Frontiers}
}
```
