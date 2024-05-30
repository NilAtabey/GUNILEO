from gnldataloader import GNLDataLoader

def main():
    # Create the dataloaders of our project
    path_data = "data/lombardgrid_front/lombardgrid/front"
    path_labels = "data/lombardgrid_alignment/lombardgrid/alignment"

    dataLoader = GNLDataLoader(path_labels, path_data, transform=None, debug=True)

    print(dataLoader[1])

if __name__ == "__main__":
    main()