export GITOPS_REPO="https://gitee.com/ren-xieyang/kdd99-latest-model"
export CURRENT_TIME=$(date +%Y-%m.%d-%H:%M)
export JENKINS_BRANCH="Jenkins_AutoPush_dagmm"


echo "[Push] Start"
export files_changed=$(git log  -1  --name-only --pretty=format:"")
export commit_comment=$(git log  -1   --pretty=format:"%s")
export model_file="deploy/model.pth"

echo "files changed in latest commit:"
echo ${files_changed}
echo ""

export result=$(echo $files_changed | grep $model_file)

if [ "${result}" != "" ]; #如果model文件被修改了
then
    export model_dir=${CURRENT_TIME}
    export tmp_dir="../../push_tmp"
    mkdir ${tmp_dir}
    mkdir ${tmp_dir}/${model_dir}
    cp ../deploy/model.pth  ${tmp_dir}/${model_dir}/model.pth
    cd  ${tmp_dir}
    git init
    git add *
    git commit -m "${commit_comment}"
    git checkout -b ${JENKINS_BRANCH}
    
    git push ${GITOPS_REPO} ${JENKINS_BRANCH}
    echo "[Push] Success"
else
    echo "[Push] Skip(model unchanged)"
fi
