import argparse
import os
import sys
import shutil
import textwrap
import typing as tp
from argparse import RawTextHelpFormatter
from pathlib import Path

import torchaudio
import pandas as pd
from tqdm import tqdm
from speechbrain.utils.data_utils import download_file

sys.path.insert(0, '../')

from asr.logger import asr_logger

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description=textwrap.dedent(
            'LibriSpeech available parts:\n' + '\n  - '.join([''] + [k for k in URL_LINKS])
        )
    )
    parser.add_argument(
        '-p',
        '--parts',
        required=True,
        type=str,
        help=(
            'A single name or names of librispeech sets separated by commas. '
            'All possible values in the description. (Example: '
            'dev-clean,train-clean-100 or dev-clean )'
        )
    )
    parser.add_argument(
        '-d',
        '--dir',
        required=True,
        type=Path,
        help='Directory for saving datasets.'
    )
    return parser.parse_args()


def _load_part(part: str, save_dir: Path) -> None:
    part_save_dir = save_dir / part
    part_save_dir.mkdir(parents=True, exist_ok=True)

    asr_logger.info(f'Starting to download {part} part into {part_save_dir}.')

    arch_path = part_save_dir / f'{part}.tar.gz'

    download_file(URL_LINKS[part], arch_path)
    shutil.unpack_archive(arch_path, part_save_dir)

    for fpath in (part_save_dir / "LibriSpeech").iterdir():
        shutil.move(str(fpath), str(part_save_dir / fpath.name))

    os.remove(str(arch_path))
    shutil.rmtree(str(part_save_dir / "LibriSpeech"))


def _process_part(part: str, save_dir: Path) -> None:
    part_save_dir = save_dir / part
    index_path = part_save_dir / 'index.tsv'

    asr_logger.info(f'Starting to prepeare index for {part} part into {index_path}.')

    flac_dirs = set()
    for dirpath, dirnames, filenames in os.walk(str(part_save_dir)):
        if any([f.endswith(".flac") for f in filenames]):
            flac_dirs.add(dirpath)

    index = []

    for flac_dir in tqdm(
        list(flac_dirs), desc=f"Preparing librispeech folders: {part}"
    ):
        flac_dir = Path(flac_dir)
        trans_path = list(flac_dir.glob("*.trans.txt"))[0]
        with trans_path.open() as f:
            for line in f:
                f_id = line.split()[0]
                f_text = " ".join(line.split()[1:]).strip()
                flac_path = flac_dir / f"{f_id}.flac"
                t_info = torchaudio.info(str(flac_path))
                length = t_info.num_frames / t_info.sample_rate
                index.append(
                    {
                        "path": str(flac_path.absolute().resolve()),
                        "text": f_text.lower(),
                        "audio_len": length,
                    }
                )

    index_df = pd.DataFrame(index)
    index_df.to_csv(index_path, index=False, sep='\t')


def main() -> None:
    args = parse_args()

    parts = args.parts.split(',')
    save_dir = args.dir.resolve()

    for part in parts:
        asr_logger.info(f'Proccess {part} part.')
        index_path = save_dir / part / f"index.tsv"
        if index_path.exists():
            asr_logger.info(
                f'Index for {part} part exists at {index_path} skip proccessing.'
            )
            continue
        _load_part(part, save_dir)
        _process_part(part, save_dir)


if __name__ == '__main__':
    main()