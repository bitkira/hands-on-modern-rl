with open('docs/appendix_game_projects/intro.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
in_env = False
in_milestones = False

for line in lines:
    if line.startswith('### 常用 RL 环境与工具'):
        in_env = True
        in_milestones = False
    elif line.startswith('### 经典里程碑项目参考'):
        in_milestones = True
        in_env = False
    elif line.startswith('### 五、实践工具与框架'):
        line = line.replace('实践工具与框架', '实践教程与技术博客')
    
    if in_env and line.startswith('### Unity ML-Agents 入门'):
        line = line.replace('###', '####', 1)
    if in_milestones and line.startswith('### ') and not line.startswith('### 经典里程碑项目参考'):
        line = line.replace('###', '####', 1)
        
    new_lines.append(line)

with open('docs/appendix_game_projects/intro.md', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
