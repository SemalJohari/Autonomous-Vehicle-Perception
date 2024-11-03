import argparse, os
import wandb
from pathlib import Path
import torchvision.models as tvmodels
import pandas as pd
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback
from sklearn.metrics import ConfusionMatrixDisplay
from IPython.display import display, Markdown

WANDB_PROJECT = "mlops-course-001"
ENTITY = None
BDD_CLASSES = {i:c for i,c in enumerate(['background', 'road', 'traffic light', 'traffic sign', 'person', 'vehicle', 'bicycle'])}
RAW_DATA_AT = 'av-team/mlops-course-001/bdd_simple_1k'
PROCESSED_DATA_AT = 'av-team/mlops-course-001/bdd_simple_1k_split'

CLASS_INDEX = {v:k for k,v in BDD_CLASSES.items()}

def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua): return True
    else: return False

def iou_per_class(inp, targ):
    "Compute iou per class"
    iou_scores = []
    for c in range(inp.shape[0]):
        dec_preds = inp.argmax(dim=0)
        p = torch.where(dec_preds == c, 1, 0)
        t = torch.where(targ == c, 1, 0)
        c_inter = (p * t).float().sum().item()
        c_union = (p + t).float().sum().item()
        iou_scores.append(c_inter / (c_union - c_inter) if c_union > 0 else np.nan)
    return iou_scores

def create_row(sample, pred_label, prediction, class_labels):
    """"A simple function to create a row of (img, target, prediction, and scores...)"""
    (image, label) = sample
    # compute metrics
    iou_scores = iou_per_class(prediction, label)
    image = image.permute(1, 2, 0)
    row =[wandb.Image(
                image,
                masks={
                    "predictions": {
                        "mask_data": pred_label[0].numpy(),
                        "class_labels": class_labels,
                    },
                    "ground_truths": {
                        "mask_data": label.numpy(),
                        "class_labels": class_labels,
                    },
                },
            ),
            *iou_scores,
    ]
    return row

def create_iou_table(samples, outputs, predictions, class_labels):
    "Creates a wandb table with predictions and targets side by side"

    def _to_str(l):
        return [f'{str(x)} IoU' for x in l]

    items = list(zip(samples, outputs, predictions))

    table = wandb.Table(
        columns=["Image"]
        + _to_str(class_labels.values()),
    )
    # we create one row per sample
    for item in progress_bar(items):
        table.add_data(*create_row(*item, class_labels=class_labels))

    return table

def get_predictions(learner, test_dl=None, max_n=None):
    """Return the samples = (x,y) and outputs (model predictions decoded), and predictions (raw preds)"""
    test_dl = learner.dls.valid if test_dl is None else test_dl
    inputs, predictions, targets, outputs = learner.get_preds(
        dl=test_dl, with_input=True, with_decoded=True
    )
    x, y, samples, outputs = learner.dls.valid.show_results(
        tuplify(inputs) + tuplify(targets), outputs, show=False, max_n=max_n
    )
    return samples, outputs, predictions

    def value(self): return self.inter/(self.union-self.inter) if self.union > 0 else None

class MIOU(DiceMulti):
    @property
    def value(self):
        binary_iou_scores = np.array([])
        for c in self.inter:
            binary_iou_scores = np.append(binary_iou_scores, \
                                          self.inter[c]/(self.union[c]-self.inter[c]) if self.union[c] > 0 else np.nan)
        return np.nanmean(binary_iou_scores)

class IOU(DiceMulti):
    @property
    def value(self):
        c=CLASS_INDEX[self.nm]
        return self.inter[c]/(self.union[c]-self.inter[c]) if self.union[c] > 0 else np.nan

class BackgroundIOU(IOU): nm = 'background'
class RoadIOU(IOU): nm = 'road'
class TrafficLightIOU(IOU): nm = 'traffic light'
class TrafficSignIOU(IOU): nm = 'traffic sign'
class PersonIOU(IOU): nm = 'person'
class VehicleIOU(IOU): nm = 'vehicle'
class BicycleIOU(IOU): nm = 'bicycle'


class IOUMacro(DiceMulti):
    @property
    def value(self):
        c=CLASS_INDEX[self.nm]
        if c not in self.count: return np.nan
        else: return self.macro[c]/self.count[c] if self.count[c] > 0 else np.nan

    def reset(self): self.macro,self.count = {},{}

    def accumulate(self, learn):
        pred,targ = learn.pred.argmax(dim=self.axis), learn.y
        for c in range(learn.pred.shape[self.axis]):
            p = torch.where(pred == c, 1, 0)
            t = torch.where(targ == c, 1, 0)
            c_inter = (p*t).float().sum(dim=(1,2))
            c_union = (p+t).float().sum(dim=(1,2))
            m = c_inter / (c_union - c_inter)
            macro = m[~torch.any(m.isnan())]
            count = macro.shape[1]

            if count > 0:
                msum = macro.sum().item()
                if c in self.count:
                    self.count[c] += count
                    self.macro[c] += msum
                else:
                    self.count[c] = count
                    self.macro[c] = msum


class MIouMacro(IOUMacro):
    @property
    def value(self):
        binary_iou_scores = np.array([])
        for c in self.count:
            binary_iou_scores = np.append(binary_iou_scores, self.macro[c]/self.count[c] if self.count[c] > 0 else np.nan)
        return np.nanmean(binary_iou_scores)


class BackgroundIouMacro(IOUMacro): nm = 'background'
class RoadIouMacro(IOUMacro): nm = 'road'
class TrafficLightIouMacro(IOUMacro): nm = 'traffic light'
class TrafficSignIouMacro(IOUMacro): nm = 'traffic sign'
class PersonIouMacro(IOUMacro): nm = 'person'
class VehicleIouMacro(IOUMacro): nm = 'vehicle'
class BicycleIouMacro(IOUMacro): nm = 'bicycle'


def display_diagnostics(learner, dls=None, return_vals=False):
    """
    Display a confusion matrix for the unet learner.
    If `dls` is None it will get the validation set from the Learner

    You can create a test dataloader using the `test_dl()` method like so:
    >> dls = ... # You usually create this from the DataBlocks api, in this library it is get_data()
    >> tdls = dls.test_dl(test_dataframe, with_labels=True)

    See: https://docs.fast.ai/tutorial.pets.html#adding-a-test-dataloader-for-inference

    """
    probs, targs = learner.get_preds(dl = dls)
    preds = probs.argmax(dim=1)
    classes = list(BDD_CLASSES.values())
    y_true = targs.flatten().numpy()
    y_pred = preds.flatten().numpy()

    tdf, pdf = [pd.DataFrame(r).value_counts().to_frame(c) for r,c in zip((y_true, y_pred) , ['y_true', 'y_pred'])]
    countdf = tdf.join(pdf, how='outer').reset_index(drop=True).fillna(0).astype(int).rename(index=BDD_CLASSES)
    countdf = countdf/countdf.sum()
    display(Markdown('### % Of Pixels In Each Class'))
    display(countdf.style.format('{:.1%}'))


    disp = ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred,
                                                   display_labels=classes,
                                                   normalize='pred')
    fig = disp.ax_.get_figure()
    fig.set_figwidth(10)
    fig.set_figheight(10)
    disp.ax_.set_title('Confusion Matrix (by Pixels)', fontdict={'fontsize': 32, 'fontweight': 'medium'})
    fig.show()

    if return_vals: return countdf, disp

default_config = SimpleNamespace(
    framework="fastai",
    img_size=180, 
    batch_size=8, 
    augment=True, 
    epochs=10,
    lr=2e-3,
    pretrained=True,  
    mixed_precision=True, 
    arch="resnet18",
    seed=42,
    log_preds=False,
)


def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--img_size', type=int, default=default_config.img_size, help='image size')
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='batch size')
    argparser.add_argument('--epochs', type=int, default=default_config.epochs, help='number of training epochs')
    argparser.add_argument('--lr', type=float, default=default_config.lr, help='learning rate')
    argparser.add_argument('--arch', type=str, default=default_config.arch, help='timm backbone architecture')
    argparser.add_argument('--augment', type=t_or_f, default=default_config.augment, help='Use image augmentation')
    argparser.add_argument('--seed', type=int, default=default_config.seed, help='random seed')
    argparser.add_argument('--log_preds', type=t_or_f, default=default_config.log_preds, help='log model predictions')
    argparser.add_argument('--pretrained', type=t_or_f, default=default_config.pretrained, help='Use pretrained model')
    argparser.add_argument('--mixed_precision', type=t_or_f, default=default_config.mixed_precision, help='use fp16')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return

def download_data():
    "Grab dataset from artifact"
    processed_data_at = wandb.use_artifact(f'{PROCESSED_DATA_AT}:latest')
    processed_dataset_dir = Path(processed_data_at.download())
    return processed_dataset_dir

def label_func(fname):
    return (fname.parent.parent/"labels")/f"{fname.stem}_mask.png"
        
def get_df(processed_dataset_dir, is_test=False):
    df = pd.read_csv(processed_dataset_dir / 'data_split.csv')
    
    if not is_test:
        df = df[df.Stage != 'test'].reset_index(drop=True)
        df['is_valid'] = df.Stage == 'valid'
    else:
        df = df[df.Stage == 'test'].reset_index(drop=True)

    df["image_fname"] = [processed_dataset_dir/f'images/{f}' for f in df.File_Name.values]
    df["label_fname"] = [label_func(f) for f in df.image_fname.values]
    return df

def get_data(df, bs=4, img_size=180, augment=True):
    block = DataBlock(blocks=(ImageBlock, MaskBlock(codes=BDD_CLASSES)),
                  get_x=ColReader("image_fname"),
                  get_y=ColReader("label_fname"),
                  splitter=ColSplitter(),
                  item_tfms=Resize((img_size, int(img_size * 16 / 9))),
                  batch_tfms=aug_transforms() if augment else None,
                 )
    return block.dataloaders(df, bs=bs)


def log_predictions(learn):
    "Log a Table with model predictions and metrics"
    samples, outputs, predictions = get_predictions(learn)
    table = create_iou_table(samples, outputs, predictions, BDD_CLASSES)
    wandb.log({"pred_table":table})
    
def final_metrics(learn):
    "Log latest metrics values"
    scores = learn.validate()
    metric_names = ['final_loss'] + [f'final_{x.name}' for x in learn.metrics]
    final_results = {metric_names[i] : scores[i] for i in range(len(scores))}
    for k,v in final_results.items(): 
        wandb.summary[k] = v

def train(config):
    set_seed(config.seed)
    run = wandb.init(project=WANDB_PROJECT, entity=ENTITY, job_type="training", config=config)
        
    config = wandb.config

    processed_dataset_dir = download_data()
    proc_df = get_df(processed_dataset_dir)
    dls = get_data(proc_df, bs=config.batch_size, img_size=config.img_size, augment=config.augment)

    metrics = [MIOU(), BackgroundIOU(), RoadIOU(), TrafficLightIOU(),
               TrafficSignIOU(), PersonIOU(), VehicleIOU(), BicycleIOU()]

    cbs = [WandbCallback(log_preds=False, log_model=True), 
           SaveModelCallback(fname=f'run-{wandb.run.id}-model', monitor='miou')]
    cbs += ([MixedPrecision()] if config.mixed_precision else [])

    learn = unet_learner(dls, arch=getattr(tvmodels, config.arch), pretrained=config.pretrained, 
                         metrics=metrics)

    learn.fit_one_cycle(config.epochs, config.lr, cbs=cbs)
    if config.log_preds:
        log_predictions(learn)
    final_metrics(learn)
    
    wandb.finish()

if __name__ == '__main__':
    parse_args()
    train(default_config)

