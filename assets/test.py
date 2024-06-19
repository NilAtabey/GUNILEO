from gnldataloader import GNLDataLoader

def main(user: int = 0):
    # Create the dataloaders of our project
    if user == 0:       # Codespace
        path_data = "/workspaces/GUNILEO/data/matching/fronts" # "data/lombardgrid_front/lombardgrid/front"
        path_labels = "/workspaces/GUNILEO/data/matching/labels" # "data/lombardgrid_alignment/lombardgrid/alignment"
    elif user == 1:     # Leo
        path_data = "data/matching/fronts" # "data/lombardgrid_front/lombardgrid/front"
        path_labels = "data/matching/labels" # "data/lombardgrid_alignment/lombardgrid/alignment"


    dataLoader = GNLDataLoader(path_labels, path_data, transform=None, debug=True)

    print(dataLoader[1:3])

if __name__ == "__main__":
    main()