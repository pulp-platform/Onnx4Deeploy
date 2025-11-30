# If you come from bash you might have to change your $PATH.
# export PATH=$HOME/bin:$HOME/.local/bin:/usr/local/bin:$PATH

# Path to your Oh My Zsh installation.
export ZSH="$HOME/.oh-my-zsh"

# Set name of the theme to load --- if set to "random", it will
# load a random theme each time Oh My Zsh is loaded, in which case,
# to know which specific one was loaded, run: echo $RANDOM_THEME
# See https://github.com/ohmyzsh/ohmyzsh/wiki/Themes
ZSH_THEME="robbyrussell"

# Set list of themes to pick from when loading at random
# Setting this variable when ZSH_THEME=random will cause zsh to load
# a theme from this variable instead of looking in $ZSH/themes/
# If set to an empty array, this variable will have no effect.
# ZSH_THEME_RANDOM_CANDIDATES=( "robbyrussell" "agnoster" )

# Uncomment the following line to use case-sensitive completion.
# CASE_SENSITIVE="true"

# Uncomment the following line to use hyphen-insensitive completion.
# Case-sensitive completion must be off. _ and - will be interchangeable.
# HYPHEN_INSENSITIVE="true"

# Uncomment one of the following lines to change the auto-update behavior
# zstyle ':omz:update' mode disabled  # disable automatic updates
# zstyle ':omz:update' mode auto      # update automatically without asking
# zstyle ':omz:update' mode reminder  # just remind me to update when it's time

# Uncomment the following line to change how often to auto-update (in days).
# zstyle ':omz:update' frequency 13

# Uncomment the following line if pasting URLs and other text is messed up.
# DISABLE_MAGIC_FUNCTIONS="true"

# Uncomment the following line to disable colors in ls.
# DISABLE_LS_COLORS="true"

# Uncomment the following line to disable auto-setting terminal title.
# DISABLE_AUTO_TITLE="true"

# Uncomment the following line to enable command auto-correction.
# ENABLE_CORRECTION="true"

# Uncomment the following line to display red dots whilst waiting for completion.
# You can also set it to another string to have that shown instead of the default red dots.
# e.g. COMPLETION_WAITING_DOTS="%F{yellow}waiting...%f"
# Caution: this setting can cause issues with multiline prompts in zsh < 5.7.1 (see #5765)
# COMPLETION_WAITING_DOTS="true"

# Uncomment the following line if you want to disable marking untracked files
# under VCS as dirty. This makes repository status check for large repositories
# much, much faster.
# DISABLE_UNTRACKED_FILES_DIRTY="true"

# Uncomment the following line if you want to change the command execution time
# stamp shown in the history command output.
# You can set one of the optional three formats:
# "mm/dd/yyyy"|"dd.mm.yyyy"|"yyyy-mm-dd"
# or set a custom format using the strftime function format specifications,
# see 'man strftime' for details.
# HIST_STAMPS="mm/dd/yyyy"

# Would you like to use another custom folder than $ZSH/custom?
# ZSH_CUSTOM=/path/to/new-custom-folder

# Which plugins would you like to load?
# Standard plugins can be found in $ZSH/plugins/
# Custom plugins may be added to $ZSH_CUSTOM/plugins/
# Example format: plugins=(rails git textmate ruby lighthouse)
# Add wisely, as too many plugins slow down shell startup.
plugins=(git)

source $ZSH/oh-my-zsh.sh

# User configuration

# export MANPATH="/usr/local/man:$MANPATH"

# You may need to manually set your language environment
# export LANG=en_US.UTF-8

# Preferred editor for local and remote sessions
# if [[ -n $SSH_CONNECTION ]]; then
#   export EDITOR='vim'
# else
#   export EDITOR='nvim'
# fi

# Compilation flags
# export ARCHFLAGS="-arch $(uname -m)"

# Set personal aliases, overriding those provided by Oh My Zsh libs,
# plugins, and themes. Aliases can be placed here, though Oh My Zsh
# users are encouraged to define aliases within a top-level file in
# the $ZSH_CUSTOM folder, with .zsh extension. Examples:
# - $ZSH_CUSTOM/aliases.zsh
# - $ZSH_CUSTOM/macos.zsh
# For a full list of active aliases, run `alias`.
#
# Example aliases
# alias zshconfig="mate ~/.zshrc"

run_cct_test() {
    local use_redmule=false
    local timestamp=$(date +%s)
    local time_mangler=$((timestamp % 10 + 1))
    local target_folder="cct_${timestamp}"
    local source_path="/app/Onnx4Deeploy/Tests/Models/CCT/onnx/CCT_train_32_128_1_2"
    local target_path="/app/Deeploy/DeeployTest/Tests/testTrainCCT/${target_folder}"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -r|--redmule)
                use_redmule=true
                echo "🔴 Redmule mode enabled"
                shift
                ;;
            -h|--help)
                echo "Usage: run_cct_test [options]"
                echo "Options:"
                echo "  -r, --redmule    Use Redmule test runner"
                echo "  -h, --help       Show this help message"
                return 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use -h or --help for usage information"
                return 1
                ;;
        esac
    done
    
    # Select the appropriate test runner and adjust folder name
    if [[ "$use_redmule" == true ]]; then
        local test_runner="testRunner_tiled_siracusa_w_redmule.py"
        target_folder="cct_redmule_${timestamp}"
        target_path="/app/Deeploy/DeeployTest/Tests/testTrainCCT/${target_folder}"
        echo "🔴 Using Redmule test runner: $test_runner"
    else
        local test_runner="testRunner_tiled_siracusa.py"
        echo "🔵 Using standard test runner: $test_runner"
    fi
    
    echo "Timestamp: $timestamp, Mangler: $time_mangler, Target folder: $target_folder"
    
    # 1. Copy from source to target
    echo "1. Copying files..."
    mkdir -p /app/Deeploy/DeeployTest/Tests/testTrainCCT/
    rm -rf "$target_path"
    
    if cp -r "$source_path" "$target_path"; then
        echo "✓ Files copied successfully to $target_folder"
    else
        echo "✗ Copy failed"
        return 1
    fi
    
    # 2. Run the test
    echo "2. Running test with $test_runner..."
    mkdir -p /app/reports/current_test
    cd /app/Deeploy/DeeployTest
    
    python "$test_runner" -t "Tests/testTrainCCT/${target_folder}" \
        --defaultMemLevel L3 \
        --l1 144000 \
        --cores 8 \
        --doublebuffer \
        --profileTiling > /app/reports/current_test/latency.txt
    
    # 3. Analyze results - 修正这里的路径逻辑
    echo "3. Analyzing results..."
    cd /app/Onnx4Deeploy/utils
    
    # 关键修正：根据使用的test runner选择正确的TEST目录
    if [[ "$use_redmule" == true ]]; then
        local test_dir="TEST_SIRACUSA_W_REDMULE"
        echo "🔴 Using Redmule test directory: $test_dir"
    else
        local test_dir="TEST_SIRACUSA"
        echo "🔵 Using standard test directory: $test_dir"
    fi
    
    local memory_html_path="/app/Deeploy/DeeployTest/${test_dir}/Tests/testTrainCCT/${target_folder}/deeployStates/memory_alloc.html"
    echo "📄 Memory allocation HTML path: $memory_html_path"
    
    if [[ -f "$memory_html_path" ]]; then
        python htmlanalyzer.py "$memory_html_path" > /app/reports/current_test/memory_alloc_report.txt
        echo "✓ Memory allocation analysis completed"
    else
        echo "⚠️  Memory allocation HTML not found at: $memory_html_path"
        echo "Available files in deeployStates directory:"
        ls -la "/app/Deeploy/DeeployTest/${test_dir}/Tests/testTrainCCT/${target_folder}/deeployStates/" 2>/dev/null || echo "Directory not found"
    fi
    
    # 4. Generate CSV report
    echo "4. Generating CSV report..."
    python /app/reports/report2csv_summary.py /app/reports/current_test/latency.txt /app/reports/current_test/latency_report.csv > /app/reports/current_test/latency_report_summary.txt
    
    echo "🎉 Test completed! Results saved in /app/reports/current_test/"
    echo "📊 Main report: latency_report.csv"
    echo "📄 Summary: latency_report_summary.txt"
    echo "💾 Memory report: memory_alloc_report.txt"
}
    
   
# Enhanced aliases
alias jump_cct_source='
cd /app/Onnx4Deeploy/Tests/Models/CCT && \
echo "Jumped to CCT source: $(pwd)"
'

alias run_cct='run_cct_test'
alias run_cct_redmule='run_cct_test --redmule'


alias goto_reports='
cd /app/reports/current_test && \
echo "📁 Current location: $(pwd)" && \
echo "📊 Available reports:" && \
ls -la
'

# Help function
cct_help() {
    echo "🚀 CCT Test Commands:"
    echo ""
    echo "Basic Usage:"
    echo "  run_cct_test              # Run standard CCT test"
    echo "  run_cct_test --redmule    # Run CCT test with Redmule"
    echo "  run_cct_test --help       # Show help"
    echo ""
    echo "Quick Aliases:"
    echo "  cct_std                   # Run standard test"
    echo "  cct_red                   # Run redmule test"
    echo "  run_cct                   # Same as run_cct_test"
    echo "  run_cct_redmule          # Same as run_cct_test --redmule"
    echo ""
    echo "Navigation:"
    echo "  jump_cct_source          # Go to CCT source directory"
    echo "  goto_reports             # Go to current test reports"
    echo ""
    echo "Analysis:"
    echo "  view_latest_report       # View latest test results"
    echo "  compare_cct_results      # Run both tests and compare"
    echo ""
    echo "🔵 Standard mode: Uses testRunner_tiled_siracusa.py"
    echo "🔴 Redmule mode: Uses testRunner_tiled_siracusa_w_Redmule.py"
}
# alias ohmyzsh="mate ~/.oh-my-zsh"

alias jump_cct_source='
cd /app/Onnx4Deeploy/Tests/Models/CCT && \
echo "Jumped to CCT source: $(pwd)" 
'