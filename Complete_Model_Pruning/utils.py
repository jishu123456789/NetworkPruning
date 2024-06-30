import torch
import torchvision.transforms as transforms
from PIL import Image
from get_loader import get_loader
import random
import os
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def caption_generate(model,dataset,image,device,max_length = 50):
    outputs=[dataset.vocab.stoi["<SOS>"]]
    for i in range(max_length):
        trg_tensor =torch.LongTensor(outputs).unsqueeze(1).to(device)
        image = image.to(device)
        
        with torch.no_grad():
            output , _  = model(image,trg_tensor)
            
        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)
        
        if best_guess == dataset.vocab.stoi["<EOS>"]:
            break
    caption = [dataset.vocab.itos[idx] for idx in outputs]
    
    return caption[1:]

def print_examples(model, device, validation_dataset):
 
    transform = transforms.Compose([transforms.Resize((350,350)),
                               transforms.RandomCrop((256,256)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    model.eval()
    
    # Now load some examples and then try to print them
    images_path , caption_path = r"C:\Users\jishu\Documents\Changwon\ImageCaptioningTrial\Image-Captioning-with-Transformers-main\flickr8k\Images" , r"C:\Users\jishu\Documents\Changwon\ImageCaptioningTrial\Image-Captioning-with-Transformers-main\flickr8k\captions.txt"

    
    validation_dataloader , _ = get_loader(images_path, caption_path, transform , train = False)
    num_samples = len(validation_dataset)
    
    random_indices = random.sample(range(num_samples), 2)
    
    for idx in random_indices:
        _ , caption = validation_dataset[idx]
        image_id = validation_dataset.imgs[idx]
        image=transform(Image.open(os.path.join(images_path,image_id)).convert("RGB")).unsqueeze(0)
        
        # Now we have to send the image to the generate caption function and the get a predicted output
        correct_label = ""
        correct_caption = caption.detach().cpu().numpy()
        
        for i in correct_caption:
            token = validation_dataset.vocab.itos[i]
            correct_label += token
            correct_label += " " 
            
            if i == 2:
                break 
        
        print("Example 1 CORRECT: " , correct_label)
        print('\n')
        print(
            "Example 1 OUTPUT: "
            + " ".join(caption_generate(model,validation_dataset,image.to(device),device,max_length = 50))
        )
    
    print("\n")
    
    test_img4 = transform(Image.open(r"C:\Users\jishu\Documents\Changwon\ImageCaptioningTrial\Image-Captioning-with-Transformers-main\Changwon.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 3 CORRECT: No correct prediction.")
    print(
        "Example 3 OUTPUT: "
        + " ".join(caption_generate(model,validation_dataset,test_img4.to(device),device,max_length = 50))
    )
    print("\n")
    model.train()


def save_checkpoint(state, epoch , filename="C:\\Users\\jishu\\Documents\\Changwon\\ImageCaptioningTrial\\Image-Captioning-with-Transformers-main\\Pruning_Checkpoints\\my_checkpoint.pth"):
    filename = Path(os.path.join(filename, f'{epoch}'))
    print("=> Saving checkpoint")
    torch.save(state, filename)
    print("=> Checkpoint saved successfully")


def load_checkpoint(checkpoint_path, model, optimizer):
    print("=> Loading checkpoint")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            step = checkpoint["step"]
            return step
        else:
            raise ValueError("Checkpoint is not a dictionary.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None