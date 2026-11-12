"""Check the graphics environment actually passed to the playback process."""

import json
import os
import shutil
import subprocess
from pathlib import Path


def test_playback_defaults_to_nvidia_rendering(tmp_path):
  scripts = tmp_path / "scripts"
  scripts.mkdir()
  source = Path(__file__).resolve().parents[1] / "scripts/play_g1_40k.sh"
  shutil.copy2(source, scripts / source.name)
  binary = tmp_path / ".venv/bin/python"
  binary.parent.mkdir(parents=True)
  binary.write_text(
    "#!/usr/bin/env python3\n"
    "import json,os,sys\n"
    "print(json.dumps({'software': os.environ.get('LIBGL_ALWAYS_SOFTWARE'),"
    "'vendor': os.environ.get('__GLX_VENDOR_LIBRARY_NAME'), 'argv': sys.argv[1:]}))\n"
  )
  binary.chmod(0o755)
  env = os.environ.copy()
  env.pop("LIBGL_ALWAYS_SOFTWARE", None)
  env.pop("__GLX_VENDOR_LIBRARY_NAME", None)
  result = subprocess.run(
    ["bash", str(scripts / source.name)],
    env=env,
    capture_output=True,
    text=True,
    check=True,
  )
  child = json.loads(result.stdout)
  assert child["software"] == "0", "Default playback must not force CPU rendering"
  assert child["vendor"] == "nvidia"
  assert child["argv"][-4:] == ["--viewer", "native", "--device", "cuda:0"]
