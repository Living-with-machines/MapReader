"""
This file is simply to avoid git conflicts. It is NOT ready to use.

Before using it:

- [ ] replace the `test_sheet_downloader.sheet_downloader` fixture with the fixture below
- [ ] create the "sample_downloaded_files", and populate it with the relevant files
- [ ] VERY IMPORTANT: move the `tile_loading._trigger_download` function, to be a static method of the `TileDownloader` class.

"""
from pathlib import Path
import shutil
from mapreader import SheetDownloader

def _mock_trigger_download(url: str, file_path: str):
    """
    This helpful functions mocks the `_trigger_download` function in tile_loading.py.
    """
    # Assume that we already have a copy of the files in the sample_downloaded_files directory
    pre_fake_download_path = Path(__file__).resolve().parent / "sample_downloaded_files"

    # Get the filename for the destination file
    fname = Path(file_path).name

    # Instead of downloading, we simply copy the file into the relevant download directory.
    shutil.copy(f"{pre_fake_download_path}/{fname}", file_path)


@pytest.fixture
def sheet_downloader(monkeypatch, sample_dir):
    test_json = f"{sample_dir}/test_json.json" # contains 4 features, 2x one-inch metadata in idx 0-1 and 2x six-inch metadata in idx 2-3
    download_url = "https://geo.nls.uk/maps/os/1inch_2nd_ed/{z}/{x}/{y}.png"
    sdl = SheetDownloader(test_json, download_url)

    monkeypatch.setattr(sdl.downloader, "_trigger_download", _mock_trigger_download)
    return sdl

