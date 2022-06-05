from distutils.command.config import config
from pathlib import Path

from loguru import logger
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from dataloader import WhaleDoDataset
from utils import *

config = {
    'csv_path': 'data/metadata.csv',
    'root_dir': 'data/',

    'dataset' : {
        'channels' : 4,
        'height': None,
        'width': None,
        'mean': None,
        'std': None,
    },

    'backbone' : {
        'model': 'resnet18',
        'rep_dim': 512,
        'pretrained': True,
    },

    'projector' : {
        'hidden_dim': 1024,
        'output_dim': 2
    },

    'batch_size': 32,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_epochs': 1000,
    'margin': 0.4,
    'save_every': 100,
    'lr': 0.001,
    'model_save_dir': 'models/',
    'model_save_name': 'whaledo_model_{}.pth',
}
config['dataset']['height'], config['dataset']['width'] = get_avg_height_width(None)
# get the mean and std of the dataset
config['dataset']['mean'], config['dataset']['std'] = get_mean_and_std_of_dataset(None)

ROOT_DIRECTORY = Path("/code_execution")
PREDICTION_FILE = ROOT_DIRECTORY / "submission" / "submission.csv"
DATA_DIRECTORY = ROOT_DIRECTORY / config['root_dir']


logger.info("Starting main script")
# load test set data and pretrained model
query_scenarios = pd.read_csv(DATA_DIRECTORY / "query_scenarios.csv", index_col="scenario_id")
metadata, _ = load_csv_and_parse_dataframe(ROOT_DIRECTORY / config['csv_path'], root_dir=DATA_DIRECTORY)
logger.info("Loading pre-trained model")

model = torch.load("model.pth", map_location=config['device']).to(config['device'])

# we'll only precompute embeddings for the images in the scenario files (rather than all images), so that the
# benchmark example can run quickly when doing local testing. this subsetting step is not necessary for an actual
# code submission since all the images in the test environment metadata also belong to a query or database.
scenario_imgs = []
for row in query_scenarios.itertuples():
    scenario_imgs.extend(pd.read_csv(DATA_DIRECTORY / row.queries_path).query_image_id.values)
    scenario_imgs.extend(pd.read_csv(DATA_DIRECTORY / row.database_path).database_image_id.values)
scenario_imgs = sorted(set(scenario_imgs))
metadata = metadata.loc[scenario_imgs]


# instantiate dataset/loader and generate embeddings for all images
dataset = WhaleDoDataset(metadata, config, mode='runtime')
dataloader = DataLoader(dataset, config['batch_size'], shuffle=False)
embeddings = []
model.eval()

logger.info("Precomputing embeddings")
for batch in tqdm(dataloader, total=len(dataloader)):
    batch_embeddings = model(batch['image'].to(config['device']))
    batch_embeddings_df = pd.DataFrame(batch_embeddings.detach().numpy(), index=batch["image_id"])
    embeddings.append(batch_embeddings_df)

embeddings = pd.concat(embeddings)
logger.info(f"Precomputed embeddings for {len(embeddings)} images")

logger.info("Generating image rankings")
# process all scenarios
results = []
for row in query_scenarios.itertuples():
    # load query df and database images; subset embeddings to this scenario's database
    qry_df = pd.read_csv(DATA_DIRECTORY / row.queries_path)
    db_img_ids = pd.read_csv(DATA_DIRECTORY / row.database_path).database_image_id.values
    db_embeddings = embeddings.loc[db_img_ids]

    # predict matches for each query in this scenario
    for qry in qry_df.itertuples():
        # get embeddings; drop query from database, if it exists
        qry_embedding = embeddings.loc[[qry.query_image_id]]
        _db_embeddings = db_embeddings.drop(qry.query_image_id, errors='ignore')

        # compute cosine similarities and get top 20
        sims = cosine_similarity(qry_embedding, _db_embeddings)[0]
        top20 = pd.Series(sims, index=_db_embeddings.index).sort_values(0, ascending=False).head(20)

        # append result
        qry_result = pd.DataFrame(
            {"query_id": qry.query_id, "database_image_id": top20.index, "score": top20.values}
        )
        results.append(qry_result)

logger.info(f"Writing predictions file to {PREDICTION_FILE}")
submission = pd.concat(results)
submission.to_csv(PREDICTION_FILE, index=False)