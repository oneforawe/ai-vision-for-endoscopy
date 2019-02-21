Some large files are tracked in this repo using git-lfs (Git Large File Storage).

Some relevant links:
* https://git-lfs.github.com
* https://github.com/git-lfs/git-lfs#getting-started

In my AWS linux instance, to get started I had to use:
* `curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash`
* `sudo apt install git-lfs`
* `git lfs install`

In my local Mac darwin system, to get started I had to use:
* `brew install git-lfs`
* `git lfs install`

Note that `git lfs install` must be run once for *every* repo instance, in a repo directory.

