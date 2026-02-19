from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.transforms.functional import to_pil_image

ORIGINAL_IMAGE = "/Users/luca/Projects/ms-data-science/deep-learning/hw11/mrshout.png"
TARGET_CLASS = "pizza"
TORCH_DEVICE = "cpu"
STEP_SIZE = 0.1


def loadimage2tensor(
    nm, resize=300, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
):
    origimage = Image.open(nm).convert("RGB")
    origimage = transforms.Resize(resize)(origimage)
    origimage = transforms.ToTensor()(origimage)
    normimage = transforms.Normalize(mean, std)(origimage)
    normimage = normimage.unsqueeze(0)  # add batch dimension

    # prüfen dass das bild gleich ist, wenn die normalisierung rückgängig gemacht wird
    std_tensor = torch.tensor(std).view(-1, 1, 1)
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    tmpimg = normimage.squeeze(0) * std_tensor + mean_tensor
    tmpimg = tmpimg
    if not torch.allclose(origimage, tmpimg):
        print("Warning image does not survive mean/std reconstruction")

    return normimage


def adv_attack(model, imorig, targetclassname, cls, stepsize):
    targetclass = cls.index(targetclassname)
    tobechanged = imorig.clone().detach().requires_grad_(True)

    currentprediction = -1
    iteration = 0

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    while currentprediction != targetclass:
        outputs = model(tobechanged)

        score = outputs[0, targetclass]
        model.zero_grad()
        score.backward()

        with torch.no_grad():
            tobechanged.data += stepsize * tobechanged.grad.sign()

            # Nach jedem Schritt das Bild wieder in den gültigen Bereich bringen
            unscaled = tobechanged.data * std + mean  # invert_normalize()
            unscaled.clamp_(0, 1)  # clamp für richtigen bereich
            tobechanged.data = (
                unscaled - mean
            ) / std  # wieder normalisieren für das modell

        tobechanged.grad.zero_()

        # Status prüfen
        with torch.no_grad():
            new_outputs = model(tobechanged)
            _, preds = torch.max(new_outputs.data, 1)
            currentprediction = preds[0].item()

            print(
                f"Iter {iteration}: Score Target: {new_outputs[0, targetclass].item():.2f}, "
                f"Pred: {cls[currentprediction]}"
            )

        iteration += 1

    # auf finales bild wieder invert_normalize anwenden
    with torch.no_grad():
        final_img = tobechanged * std + mean
        return to_pil_image(final_img.squeeze(0).clamp(0, 1))


# Load model
device = torch.device(TORCH_DEVICE)
weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1.DEFAULT
cls = weights.meta["categories"]
preprocess = weights.transforms()
model = efficientnet_v2_s(weights=weights)
model.to(device)
model.eval()

# Load original image
imgorig = loadimage2tensor(ORIGINAL_IMAGE)

# Run adv attack
imgadv = adv_attack(model, imgorig, TARGET_CLASS, cls, STEP_SIZE)
imgadv.save("output.png")
