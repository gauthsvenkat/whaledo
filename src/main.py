from distutils.command.config import config
from pathlib import Path

from sklearn import preprocessing
from config import get_config

from loguru import logger
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import WhaleDoDataset
from utils import *
import os

config = get_config()
config['dataset']['height'], config['dataset']['width'] = get_avg_height_width(None)
config['dataset']['mean'], config['dataset']['std'] = get_mean_and_std_of_dataset(None)

ROOT_DIRECTORY = Path("/code_execution")

if not os.path.exists(ROOT_DIRECTORY):
    logger.info("Not inside container. Setting ROOT_DIRECTORY to workspace directory")
    ROOT_DIRECTORY = Path("../")

PREDICTION_FILE = ROOT_DIRECTORY / "submission" / "submission.csv"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"


def setup_dataloader(metadata_df):
    # Preprocess metadata
    metadata_df['path'] = metadata_df['path'].map(lambda p: os.path.join(DATA_DIRECTORY, p)) #convert path to full path
    metadata_df['viewpoint'] = metadata_df['viewpoint'].map({'top': 0, 'left': -1, 'right': 1}) #convert viewpoint to 0, -1, 1

    # Setup dataloader
    dataset = WhaleDoDataset(metadata_df, config, mode='runtime')
    dataloader = DataLoader(dataset, config['main_batch_size'], shuffle=False)
    return dataloader

def generate_embeddings(model, dataloader):
    embeddings = []

    for batch in tqdm(dataloader, total=len(dataloader), desc="Precomputing embeddings"):
        batch_embeddings = model(batch['image'].to(config['device']))
        batch_embeddings_normed = preprocessing.normalize(batch_embeddings.detach().cpu().numpy())
        batch_embeddings_df = pd.DataFrame(batch_embeddings_normed, index=batch["image_id"])
        embeddings.append(batch_embeddings_df)

    embeddings = pd.concat(embeddings)
    logger.info(f"Precomputed embeddings for {len(embeddings)} images")
    return embeddings


def predict(qry_embedding, db_embeddings):
    # Compute euclidean distance between all images
    distances = euclidean_distances(qry_embedding, db_embeddings)[0]
    # Turn distances into similarity scores
    sims = map(lambda d: 1/(1+d), distances)
    # Select top 20 pairs
    top20 = pd.Series(sims, index=db_embeddings.index).sort_values(0, ascending=False).head(20)
    result_images, scores = top20.index, top20.values

    return result_images, scores


def main():
    logger.info("Starting main script")

    # Load scenarios and metadata
    scenarios_df = pd.read_csv(DATA_DIRECTORY / "query_scenarios.csv")
    metadata_df = pd.read_csv(DATA_DIRECTORY / "metadata.csv", index_col='image_id')

    # Setup dataloader
    dataloader = setup_dataloader(metadata_df)

    # Load model
    model = torch.load("model.pth", map_location=config['device']).to(config['device'])
    model.projector = None # remove projector during testing
    model.eval()

    # Generate embeddings
    embeddings = generate_embeddings(model, dataloader)

    # Process all scenarios
    predictions = []
    for scenario_row in scenarios_df.itertuples():

        # load query df and database images; subset embeddings to this scenario's database
        queries_df = pd.read_csv(DATA_DIRECTORY / scenario_row.queries_path)
        database_df = pd.read_csv(DATA_DIRECTORY / scenario_row.database_path)
        database_image_ids = database_df["database_image_id"].values
        database_embeddings = embeddings.loc[database_image_ids]
        
        # predict matches for each query in this scenario
        for query_row in queries_df.itertuples():
            query_id = query_row.query_id
            query_image_id = query_row.query_image_id

            # get embeddings; drop query from database, if it exists
            qry_embedding = embeddings.loc[[query_image_id]]
            _db_embeddings = database_embeddings.drop(query_image_id, errors='ignore')
            result_images, scores = predict(qry_embedding, _db_embeddings)

            # Append result
            prediction = pd.DataFrame({"query_id": query_id, "database_image_id": result_images, "score": scores})
            predictions.append(prediction)

    predictions_df = pd.concat(predictions)
    predictions_df.to_csv(PREDICTION_FILE, index=False)


if __name__ == "__main__":
    main()
