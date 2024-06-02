from gnldataloader import GNLDataLoader

def main():
    # Create the dataloaders of our project
    path_data = "/workspaces/GUNILEO/data/matching/fronts" # "data/lombardgrid_front/lombardgrid/front"
    path_labels = "/workspaces/GUNILEO/data/matching/labels" # "data/lombardgrid_alignment/lombardgrid/alignment"

    dataLoader = GNLDataLoader(path_labels, path_data, transform=None, debug=True)

    print(dataLoader[1:10:2])

if __name__ == "__main__":
    main()