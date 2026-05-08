## 配置文件说明

### configure_app.toml
业务能修改到的KaiwuDRL的框架配置, 由使用侧修改

## kaiwudrl
KaiwuDRL框架使用

### client.toml
client工具, 由框架侧修改

### aisrv.toml
aisrv进程, 由框架侧修改

### actor.toml
actor进程, 由框架侧修改

### learner.toml
learner进程, 由框架侧修改

### configure.toml
公共配置, 由框架侧修改


## 注意事项
kaiwudrl目录下的配置文件, 在对外时建议隐藏, 由框架侧确定默认的参数值
