from pathlib import Path
import pandas as pd

ROOT_DIRECTORY = Path("/code_execution")
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
OUTPUT_FILE = ROOT_DIRECTORY / "submission/submission.csv"


def predict(query_image_id, database_image_ids):
    """
    Predict function of our model.
    :param query_image_id: identifier of the query_image from the queries/scenario##.csv file.
        This is the image we need to match against the database.
    :param database_image_ids: identifier of the image you are returning for that query
        These are the database images of other whales.
    :return:
        result_images: set of images we predict to be the same whale.
        scores: confidence scores of how sure the model is it is the same whale.
    """
    raise NotImplementedError(
        "This script is just a template. You should adapt it with your own code."
    )
    result_images = []  # Images that we match with the query_image
    scores = []  # Confidence score, in the range [0.0, 1.0].
    return result_images, scores


def main():
    scenarios_df = pd.read_csv("/code_execution/query_scenarios.csv")
    metadata_df = pd.read_csv("/code_execution/metadata.csv")

    predictions = []

    for scenario_row in scenarios_df.itertuples():

        queries_df = pd.read_csv(DATA_DIRECTORY / scenario_row.queries_path)
        database_df = pd.read_csv(DATA_DIRECTORY / scenario_row.database_path)

        for query_row in queries_df.itertuples():
            query_id = query_row.query_id
            query_image_id = query_row.query_image_id
            database_image_ids = database_df["database_image_id"].values

            # PREDICTION HAPPENS HERE
            result_images, scores = predict(query_image_id, database_image_ids)

            for pred_image_id, score in zip(result_images, scores):
                predictions.append(
                    {
                        "query_id": query_id,
                        "database_image_id": pred_image_id,
                        "score": score,
                    }
                )

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
