# Git 命令行操作指南

> 适用项目：`MAKER_Minimal`（https://github.com/hepore/MAKER_Minimal）
> 环境：Windows + PowerShell，全部用命令行操作。

---

## 一、核心心智模型（先记住这三句）

| 动作 | 命令 | 意思 |
|---|---|---|
| 暂存 | `git add .` | 把改动放进"待提交清单" |
| 存档 | `git commit -m "说明"` | 在本地记一笔快照 |
| 上传 | `git push` | 把本地提交传到 GitHub |

- 本地改动**不会自动同步**，必须 `add → commit → push` 三步走完，GitHub 上才更新。
- `git pull` = 把 GitHub 上的新东西拉到本地（师妹 push 后，你用 `git pull` 同步）。

---

## 二、一次性配置（这台电脑只做一次）

```powershell
# 1. 身份（提交时盖的"作者章"）
git config --global user.name "hepore"
git config --global user.email "hepore@163.com"

# 2. 代理（国内访问 GitHub 需要；端口换成你代理软件的实际端口）
git config --global http.proxy http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897

# 3. 关闭证书吊销检查（否则 Windows 上可能报 CRYPT_E_REVOCATION_OFFLINE）
git config --global http.schannelCheckRevoke false
```

> 查看 / 取消代理：
> ```powershell
> git config --global --get http.proxy      # 查看
> git config --global --unset http.proxy    # 取消（不用代理时）
> git config --global --unset https.proxy
> ```

---

## 三、日常更新流程（每次改完代码就这三步）

```powershell
cd E:\PaperWorkSpace\3\Experiment\MAKER_Minimal

git status                  # （可选）先看看改了什么、当前状态
git add .                   # 1. 暂存所有改动
git commit -m "改了xxx"      # 2. 本地存档，附一句说明
git push                    # 3. 上传到 GitHub
```

> 只提交某一个文件：
> ```powershell
> git add demo.py
> git commit -m "更新 demo"
> git push
> ```

---

## 四、常用命令速查

| 命令 | 作用 |
|---|---|
| `git status` | 看当前改了哪些文件、是否已暂存 |
| `git diff` | 看具体改了哪些内容（还没 add 的） |
| `git log --oneline` | 看提交历史（每条一行的简短列表） |
| `git remote -v` | 看关联的远程仓库地址 |
| `git pull` | 从 GitHub 拉取最新代码 |
| `git branch` | 看当前在哪个分支 |

---

## 五、常见报错排查

| 报错 | 原因 | 解决 |
|---|---|---|
| `Recv failure: Connection was reset` | 直连 GitHub 被墙 / 代理没开 | 开代理软件 + 系统代理，配 `http.proxy`（见第二节） |
| `CRYPT_E_REVOCATION_OFFLINE` | 证书吊销检查失败 | `git config --global http.schannelCheckRevoke false` |
| `Authentication failed` | 密码/token 不对 | GitHub 已禁用密码 push，改用 Personal Access Token |
| `failed to push ... (non-fast-forward)` | 远程有新提交，本地落后 | 先 `git pull` 再 `git push` |

---

## 六、提交信息规范（建议）

- 一行说清楚"做了什么"：`git commit -m "修复先验组合公式"`
- 用动词开头：新增 / 修改 / 修复 / 删除 / 重构
- 别用 "update" "fix bug" 这种没信息量的描述

---

## 七、给师妹 / 新人的 clone 流程（首次拉取）

```powershell
git clone https://github.com/hepore/MAKER_Minimal.git
cd MAKER_Minimal
pip install -r requirements.txt
python demo.py
```
