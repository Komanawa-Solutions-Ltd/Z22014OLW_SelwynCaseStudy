# Z22014OLW_SelwynCaseStudy
created with api

# setup remote run

.. code-block::bash

    mkdir ~/PycharmProjects/Z22014OLW_SelwynCaseStudy
    cd ~/PycharmProjects/Z22014OLW_SelwynCaseStudy
    git clone https://$kslgittoken@github.com/Komanawa-Solutions-Ltd/Z22014OLW_SelwynCaseStudy.git


# remote run

```bash
# use tmux
#tmux new -s [session_name]
conda activate OLW2
cd ~/PycharmProjects/modflow_tools_olw
git fetch --all
git reset --hard origin/main
cd ~/PycharmProjects/Z22014OLW_SelwynCaseStudy
git fetch --all
git reset --hard origin/main
# -u for unbuffered output, note calling conda run python -u [scriptname].py does not work
PYTHONPATH="$PYTHONPATH:$PWD:~/PycharmProjects/modflow_tools_olw" python -u [scriptname].py

# disconnect tmux session
ctrl+b d

# attach to tmux session
tmux a
```