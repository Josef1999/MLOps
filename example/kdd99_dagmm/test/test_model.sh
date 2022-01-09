echo "[Test] Start"
python --version
#lsb_release -a

pip install --default-timeout=1000 -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

pytest -s --disable-warnings
export pytest_result=$?
if [ pytest_result==0 ];
then
    echo "[Test] Success"
else
    echo "[Test] Failed"
    return 1
fi