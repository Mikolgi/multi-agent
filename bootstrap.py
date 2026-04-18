from __future__ import annotations

import site
import sys


def ensure_user_site_on_path() -> None:
    user_site = site.getusersitepackages()
    if user_site and user_site not in sys.path:
        sys.path.append(user_site)
