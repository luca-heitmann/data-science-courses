import PIL.Image
from torchvision.utils import draw_bounding_boxes

from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights

from torchvision.transforms.functional import to_pil_image
import torchvision.models.detection
import torch

def run():

  box_score_thresh=0.85

  dataset_path = '/Users/luca/Projects/ms-data-science/deep-learning/hw9/objdetcodeforstudents/testimages1/'
  filename = f"{dataset_path}/morecows_4291970_960_720.jpg"

  #https://pytorch.org/vision/stable/models.html  
  modelweights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1.DEFAULT

  model = retinanet_resnet50_fpn_v2(weights=modelweights)
  
  transforms = modelweights.transforms()
  
  # prepare batch made from one image
  img = PIL.Image.open(filename).convert('RGB')  
  batch = [transforms(img)]
  
  # if you forget that, you will see a nice error :) 
  model.eval()
  
  with torch.no_grad():

    # call your model on the batch
    predictions = model(batch)
    
    pred = predictions[0] # batch has size 1, see above, so we grab that one element here

    predicted = {
      "boxes": pred["boxes"][pred["scores"] > box_score_thresh],
      "labels": pred["labels"][pred["scores"] > box_score_thresh],
      "scores": pred["scores"][pred["scores"] > box_score_thresh],
    }
    
    #get names of labels which are predicted 
    labels_in_image = [modelweights.meta["categories"][i] for i in predicted["labels"]]

  # output what is predicted to understand the format of torchvision model outputs
  # it is a list of dictionaries. list has length equal to batchsize
  print(type(predictions))
  print(type(predicted))
  # each dictionary has keys for the outputs which one would expect from an object detection neural net
  for k in predicted.keys():
    print('dictionary key name {}'.format(k))  

  print(predicted)    
  print('unique predicted labels:')
  print( set(labels_in_image))

  # draw the boundingbox
  box = draw_bounding_boxes(batch[0], boxes=predicted["boxes"],
                            labels=labels_in_image,
                            colors="red",
                            width=4 ) #, font_size=30, font =None)
  #print convert to a PIL.Image                                                      
  im = to_pil_image(box.detach())
  #display the PIL.Image
  im.show()



if __name__=='__main__':
  run()  


    
