import subprocess
import os
import shutil
import argparse

def checkout_svn_branch(url_base, branch, target, username, password, revision=None):
    """
    从 SVN 检出指定分支和修订版本的代码。

    参数:
        url_base (str): SVN 仓库根地址（不含分支部分)
        branch (str): 分支路径，例如 "branches/dev" 或 "trunk"
        target (str): 本地检出目录
        username (str): SVN 用户名
        password (str): SVN 密码
        revision (str or int, optional): 要检出的修订版本号（如 1234 或 "1234"），默认为最新
    """
    # 构建完整 URL
    if branch.strip() == "trunk":
        full_url = f"{url_base.rstrip('/')}/trunk"
    else:
        # 自动处理是否包含 branches/
        branch_clean = branch.strip().lstrip('/')
        if not branch_clean.startswith("branches/"):
            branch_clean = f"branches/{branch_clean}"
        full_url = f"{url_base.rstrip('/')}/{branch_clean}"

    # 构建命令
    cmd = [
        "svn", "checkout", full_url, target,
        "--username", username,
        "--password", password,
        "--non-interactive", "--trust-server-cert"
    ]

    if revision is not None:
        cmd.extend(["--revision", str(revision)])

    # 执行命令
    subprocess.check_call(cmd)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Checkout SVN branch with optional revision.")
    
    parser.add_argument(
        "--branch", 
        type=str, 
        required=True, 
        help="Branch name (e.g., 'dev', 'test'). Will be prefixed with 'branches/' unless 'trunk'."
    )
    parser.add_argument(
        "--revision", 
        type=int, 
        default=None, 
        help="Revision number to checkout (e.g., 7). If not provided, checks out HEAD."
    )

    parser.add_argument(
        "--user", 
        type=str, 
        default="marcusjzhou",
        help="SVN username."
    )
    parser.add_argument(
        "--passwd", 
        type=str, 
        default="123456",
        help="SVN password."
    )

    args = parser.parse_args()

    url_base = "https://svn.tencent.com/HPRC/HPRC_Test1"
    test_dir_name = "branches/" + args.branch

    checkout_svn_branch(url_base, args.branch, test_dir_name, args.user, args.passwd, revision=args.revision)

    # 使用Python复制Test.py文件到test_dir_name文件夹并覆盖
    source_file = "Test.py"
    destination_file = os.path.join(test_dir_name, "Test.py")

    for fname in ["Test.py", "Utils.py", "config.yaml"]:
        source_file = fname
        destination_file = os.path.join(test_dir_name, fname)
        try:
            shutil.copy2(source_file, destination_file)
        except FileNotFoundError:
            print(f"源文件 {source_file} 不存在")
        except Exception as e:
            print(f"复制文件时发生错误: {e}")

    # 运行测试
    try:
        subprocess.run(["python", "./Test.py"], cwd=test_dir_name, check=True)
    except subprocess.CalledProcessError as e:
        print(f"测试运行失败: {e}")
    except Exception as e:
        print(f"运行测试时发生错误: {e}")
