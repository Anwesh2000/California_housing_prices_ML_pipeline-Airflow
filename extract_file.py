import os
import tarfile
import urllib


def fetch_housing_data():
    print('Running')

    download_root = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    housing_path = r'data_file'

    housing_url = download_root + "datasets/housing/housing.tgz"
    tgz_path = os.path.join(housing_path, "housing.tgz")

    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    print('Extracted and saved housing data set')
