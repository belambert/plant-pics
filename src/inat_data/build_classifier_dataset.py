import json

from datasets import Dataset, Image


def main():
    with open ("inat_large/results.json") as f:
        data = json.load(f)

    images = ["inat_large/" + x["image"] for x in data]
    classes = [x["class"] for x in data]

    dataset_dict = {
        "image": images,
        "class": classes
    }

    ds = Dataset.from_dict(dataset_dict).cast_column("image", Image())

    print(ds[0:5])
    ds.push_to_hub("blambert/inat_sample_classes")
    # print(data[:5])

if __name__ == "__main__":
    main()
