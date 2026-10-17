from datasets import load_dataset


def data():


    ds = load_dataset("cyanic-selkie/aida-conll-yago-wikidata", split="test")
    print(ds[0])




def main():
    print("Hello from text-to-wikidata-linking!")
    data()


if __name__ == "__main__":
    main()
