rsync -avz -e /bin/ssh sim2:/home/azureuser/splitwise-sim/results/ results/
ssh sim2 "rm -rf splitwise-sim/results/*"
rsync -avz -e /bin/ssh sim3:/home/azureuser/splitwise-sim/results/ results/
ssh sim3 "rm -rf splitwise-sim/results/*"

以下是每行代码的详细解释，包括其语法和作用：

```bash
rsync -avz -e /bin/ssh sim2:/home/azureuser/splitwise-sim/results/ results/
```

- **`rsync -avz -e /bin/ssh`**：使用 `rsync` 命令通过 SSH 复制文件。
  - **`rsync`**：一种用于文件同步和传输的工具。
  - **`-a`**：开启归档模式，表示递归地复制文件夹，并保持文件的权限、时间戳、符号链接等。
  - **`-v`**：启用详细模式，显示传输过程中的信息。
  - **`-z`**：启用压缩，在传输过程中压缩文件，减少传输数据量。
  - **`-e /bin/ssh`**：指定使用 `/bin/ssh` 作为远程 shell，以确保通过 SSH 进行传输。
- **`sim2:/home/azureuser/splitwise-sim/results/`**：源目录路径，表示从远程主机 `sim2` 的 `/home/azureuser/splitwise-sim/results/` 目录中获取文件。
- **`results/`**：目标目录，将文件复制到本地的 `results/` 目录中。

此命令的作用是将远程服务器 `sim2` 上的 `results` 目录内容复制到本地的 `results/` 目录中。

---

```bash
ssh sim2 "rm -rf splitwise-sim/results/*"
```

- **`ssh sim2 "rm -rf splitwise-sim/results/*"`**：使用 SSH 登录到 `sim2` 服务器并删除指定目录中的内容。
  - **`ssh sim2`**：通过 SSH 登录到远程主机 `sim2`。
  - **`"rm -rf splitwise-sim/results/*"`**：在远程主机上执行删除命令。
    - **`rm -rf`**：删除指定的文件或目录并递归删除所有内容。
    - **`splitwise-sim/results/*`**：指定要删除的路径，`*` 表示删除该目录下的所有文件和子文件夹。

此命令的作用是在 `sim2` 服务器上删除 `splitwise-sim/results/` 目录中的所有内容，以清理该目录。

---

```bash
rsync -avz -e /bin/ssh sim3:/home/azureuser/splitwise-sim/results/ results/
```

- **`rsync -avz -e /bin/ssh`**：再次使用 `rsync` 命令通过 SSH 复制文件（详细说明如上）。
- **`sim3:/home/azureuser/splitwise-sim/results/`**：源目录路径，表示从远程主机 `sim3` 的 `/home/azureuser/splitwise-sim/results/` 目录中获取文件。
- **`results/`**：目标目录，将文件复制到本地的 `results/` 目录中。

此命令的作用是将远程服务器 `sim3` 上的 `results` 目录内容复制到本地的 `results/` 目录中。

---

```bash
ssh sim3 "rm -rf splitwise-sim/results/*"
```

- **`ssh sim3 "rm -rf splitwise-sim/results/*"`**：使用 SSH 登录到 `sim3` 服务器并删除指定目录中的内容（详细说明如上）。

此命令的作用是在 `sim3` 服务器上删除 `splitwise-sim/results/` 目录中的所有内容，以清理该目录。

---

### 总结

这些命令的整体作用是：
1. 使用 `rsync` 从远程服务器 `sim2` 和 `sim3` 上同步 `results` 目录内容到本地。
2. 在同步完成后，通过 SSH 登录到 `sim2` 和 `sim3` 服务器，并删除远程 `results` 目录的内容，以清理该目录。