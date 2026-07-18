#!/usr/bin/env node

const fs = require('fs')
const path = require('path')
const { execFileSync } = require('child_process')
const https = require('https')

const CONFIG_PATH = path.join(__dirname, 'config.json')

function readConfig() {
  if (!fs.existsSync(CONFIG_PATH)) {
    return {}
  }
  try {
    return JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8'))
  } catch (err) {
    return {}
  }
}

function parseArgs(argv) {
  const args = {
    command: argv[2],
    workspace: null,
    date: null,
    since: null,
    until: null,
    period: null
  }
  for (let i = 3; i < argv.length; i++) {
    const item = argv[i]
    if (item === '--workspace') { args.workspace = argv[i + 1]; i++ }
    if (item === '--date') { args.date = argv[i + 1]; i++ }
    if (item === '--since') { args.since = argv[i + 1]; i++ }
    if (item === '--until') { args.until = argv[i + 1]; i++ }
    if (item === '--period') { args.period = argv[i + 1]; i++ }
    if (item === '--stdin') { args.stdin = true }
  }
  return args
}

function outputJson(data) {
  process.stdout.write(JSON.stringify(data, null, 2))
}

function isRiskyWorkspace(dir) {
  const normalized = path.resolve(dir)
  const isWindows = process.platform === 'win32'
  const risky = isWindows
    ? ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'].map(d => path.resolve(`${d}:\\`))
    : [path.resolve('/'), path.resolve('/Users'), path.resolve('/home'), path.resolve('/root')]
  return risky.includes(normalized)
}

function shouldIgnoreDir(name, config) {
  const defaultIgnore = ['node_modules','.git','dist','build','coverage','.next','.nuxt','.output','.cache','.turbo','.vite','.idea','.vscode','logs','tmp','temp','target','.gradle','out']
  const userIgnore = config.ignore && Array.isArray(config.ignore.dirs) ? config.ignore.dirs : []
  return new Set([...defaultIgnore, ...userIgnore]).has(name)
}

function isGitRepo(dir) {
  return fs.existsSync(path.join(dir, '.git'))
}

function findGitRepos(workspaceDirs, config) {
  const repos = []
  const maxDepth = config.scan && config.scan.maxDepth ? config.scan.maxDepth : 3
  const maxRepos = config.scan && config.scan.maxRepos ? config.scan.maxRepos : 100
  function walk(currentDir, depth) {
    if (repos.length >= maxRepos) return
    if (!fs.existsSync(currentDir)) return
    if (isGitRepo(currentDir)) { repos.push(currentDir); return }
    if (depth >= maxDepth) return
    let entries = []
    try { entries = fs.readdirSync(currentDir, { withFileTypes: true }) } catch (err) { return }
    for (const entry of entries) {
      if (!entry.isDirectory()) continue
      if (shouldIgnoreDir(entry.name, config)) continue
      walk(path.join(currentDir, entry.name), depth + 1)
    }
  }
  for (const workspace of workspaceDirs) { walk(path.resolve(workspace), 0) }
  return repos
}

function runGit(repoPath, args) {
  try {
    return execFileSync('git', args, { cwd: repoPath, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim()
  } catch (err) {
    if (err.code === 'ENOENT') throw new Error('找不到 git 命令，请确认 git 已安装并加入 PATH 环境变量。')
    return ''
  }
}

function getBranch(repoPath) {
  return runGit(repoPath, ['rev-parse', '--abbrev-ref', 'HEAD'])
}

function formatDateLocal(date) {
  const y = date.getFullYear()
  const m = String(date.getMonth() + 1).padStart(2, '0')
  const d = String(date.getDate()).padStart(2, '0')
  return `${y}-${m}-${d}`
}

function getDateRange(cliArgs) {
  const today = new Date()
  const todayStr = formatDateLocal(today)
  if (cliArgs.since && cliArgs.until) {
    return { period: 'custom', date: todayStr, since: `${cliArgs.since} 00:00:00`, until: `${cliArgs.until} 23:59:59` }
  }
  if (cliArgs.since || cliArgs.until) {
    return { error: { code: 'MISSING_DATE_RANGE', message: '--since 和 --until 必须同时指定。' } }
  }
  const period = cliArgs.period || 'daily'
  if (period === 'weekly') {
    const dayOfWeek = today.getDay() || 7
    const monday = new Date(today)
    monday.setDate(today.getDate() - dayOfWeek + 1)
    const mondayStr = formatDateLocal(monday)
    return { period: 'weekly', date: todayStr, since: `${mondayStr} 00:00:00`, until: `${todayStr} 23:59:59` }
  }
  if (period === 'quarterly') {
    const quarter = Math.floor(today.getMonth() / 3)
    const quarterStart = new Date(today.getFullYear(), quarter * 3, 1)
    const quarterStartStr = formatDateLocal(quarterStart)
    return { period: 'quarterly', date: todayStr, since: `${quarterStartStr} 00:00:00`, until: `${todayStr} 23:59:59` }
  }
  if (period === 'yearly') {
    const yearStart = `${today.getFullYear()}-01-01`
    return { period: 'yearly', date: todayStr, since: `${yearStart} 00:00:00`, until: `${todayStr} 23:59:59` }
  }
  const date = cliArgs.date || todayStr
  return { period: 'daily', date, since: `${date} 00:00:00`, until: `${date} 23:59:59` }
}

function parseCommitType(message) {
  const match = message.match(/^(\w+)(?:\(([^)]+)\))?:\s*(.+)$/)
  if (!match) return { type: null, scope: null }
  return { type: match[1], scope: match[2] || null }
}

function getGitUserEmail() {
  try {
    return execFileSync('git', ['config', '--global', 'user.email'], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim()
  } catch { return '' }
}

function getTodayCommits(repoPath, dateRange, authorEmails) {
  const pretty = '%H%x1f%h%x1f%an%x1f%ae%x1f%ad%x1f%s%x1e'
  const args = ['log', `--since=${dateRange.since}`, `--until=${dateRange.until}`, `--pretty=format:${pretty}`, '--date=iso']
  const raw = runGit(repoPath, args)
  if (!raw) return []
  return raw.split('\x1e').map(item => item.trim()).filter(Boolean).map(item => {
    const parts = item.split('\x1f')
    const message = parts[5] || ''
    const parsed = parseCommitType(message)
    return { hash: parts[0] || '', shortHash: parts[1] || '', authorName: parts[2] || '', authorEmail: parts[3] || '', time: parts[4] || '', message, type: parsed.type, scope: parsed.scope }
  }).filter(commit => { if (!authorEmails.length) return true; return authorEmails.includes(commit.authorEmail) })
}

function getLastCommitTimeMap(repoPath) {
  const since = new Date(Date.now() - 30 * 24 * 3600 * 1000)
  const sinceStr = formatDateLocal(since)
  const raw = runGit(repoPath, ['log', '--no-merges', '--name-only', '--format=%ct', `--since=${sinceStr}`])
  if (!raw) return {}
  const map = {}
  let currentTs = null
  for (const line of raw.split('\n')) {
    const trimmed = line.trim()
    if (!trimmed) continue
    if (/^\d+$/.test(trimmed)) { currentTs = parseInt(trimmed, 10) * 1000 }
    else if (currentTs !== null) { if (!map[trimmed]) map[trimmed] = currentTs }
  }
  return map
}

function parseGitStatusLine(line) {
  const match = line.match(/^([MADRC? ])([MADRC?! ]) (.+)$/)
  if (match) {
    const status = (match[1] + match[2]).replace(/ /g, '_')
    const rawFile = match[3]
    const renameMatch = rawFile.match(/^(.+?)\s*->\s*(.+)$/)
    const file = renameMatch ? renameMatch[2].trim() : rawFile
    return { file, status }
  }
  const shortMatch = line.match(/^([MADRC?]) (.+)$/)
  if (shortMatch) {
    const status = shortMatch[1] + '_'
    const rawFile = shortMatch[2]
    const renameMatch = rawFile.match(/^(.+?)\s*->\s*(.+)$/)
    const file = renameMatch ? renameMatch[2].trim() : rawFile
    return { file, status }
  }
  return null
}

function getUncommittedChanges(repoPath, staleMaxAge) {
  const raw = runGit(repoPath, ['status', '--porcelain'])
  if (!raw) return []
  const now = Date.now()
  const cutoff = staleMaxAge > 0 ? now - staleMaxAge * 24 * 3600 * 1000 : 0
  const commitTimeMap = staleMaxAge > 0 ? getLastCommitTimeMap(repoPath) : {}
  return raw.split('\n').map(line => {
    const parsed = parseGitStatusLine(line)
    if (!parsed) return null
    const { file, status } = parsed
    let stale = false
    if (cutoff > 0 && file) {
      const isUntracked = status.includes('?')
      if (!isUntracked) {
        const lastCommitMs = commitTimeMap[file]
        if (lastCommitMs !== undefined) { stale = lastCommitMs < cutoff }
        else {
          const singleTs = runGit(repoPath, ['log', '-1', '--no-merges', '--format=%ct', '--', file])
          if (singleTs) { stale = parseInt(singleTs, 10) * 1000 < cutoff }
          else {
            try { const stat = fs.statSync(path.join(repoPath, file)); if (stat.mtimeMs < cutoff) stale = true } catch {}
          }
        }
      } else {
        try { const stat = fs.statSync(path.join(repoPath, file)); if (stat.mtimeMs < cutoff) stale = true } catch {}
      }
    }
    return { file, status, stale }
  }).filter(item => item !== null && item.file)
}

function collect(config, cliArgs) {
  const workspaceDirs = cliArgs.workspace ? [cliArgs.workspace] : Array.isArray(config.workspaceDirs) ? config.workspaceDirs : []
  if (!workspaceDirs.length) {
    return { ok: false, error: { code: 'WORKSPACE_REQUIRED', message: '未配置 workspaceDirs，请先指定扫描范围。' } }
  }
  for (const workspace of workspaceDirs) {
    if (isRiskyWorkspace(workspace)) {
      return { ok: false, error: { code: 'RISKY_WORKSPACE', message: `扫描范围过大：${workspace}。请指定更小的 workspace。` } }
    }
  }
  const dateRange = getDateRange(cliArgs)
  if (dateRange.error) return { ok: false, error: dateRange.error }
  const repos = findGitRepos(workspaceDirs, config)
  const staleMaxAge = config.report && typeof config.report.staleMaxAge === 'number' ? config.report.staleMaxAge : 7
  const configEmails = (Array.isArray(config.authorEmails) ? config.authorEmails : []).filter(e => e && e.trim())
  const defaultEmail = getGitUserEmail()
  const authorEmails = configEmails.length > 0 ? configEmails : (defaultEmail ? [defaultEmail] : [])
  const activeProjects = []
  const inactiveProjects = []
  for (const repo of repos) {
    const name = path.basename(repo)
    const branch = getBranch(repo)
    const commits = getTodayCommits(repo, dateRange, authorEmails)
    const uncommittedChanges = getUncommittedChanges(repo, staleMaxAge)
    const activeUncommittedCount = uncommittedChanges.filter(c => !c.stale).length
    const staleUncommittedCount = uncommittedChanges.filter(c => c.stale).length
    const activeReasons = []
    if (commits.length > 0) activeReasons.push('today_commits')
    if (activeUncommittedCount > 0) activeReasons.push('uncommitted_changes')
    if (staleUncommittedCount > 0) activeReasons.push('stale_changes')
    const item = { name, path: repo, branch, activeReasons, commits, uncommittedChanges, stats: { commitCount: commits.length, uncommittedFileCount: activeUncommittedCount, staleFileCount: staleUncommittedCount } }
    if (activeReasons.length > 0 && activeReasons.some(r => r !== 'stale_changes')) { activeProjects.push(item) }
    else { inactiveProjects.push(item) }
  }
  const totalCommitCount = activeProjects.reduce((sum, item) => sum + item.stats.commitCount, 0)
  const totalUncommittedFileCount = activeProjects.reduce((sum, item) => sum + item.stats.uncommittedFileCount, 0)
  const totalStaleFileCount = [...activeProjects, ...inactiveProjects].reduce((sum, item) => sum + item.stats.staleFileCount, 0)
  return {
    ok: true,
    meta: { period: dateRange.period, date: dateRange.date, since: dateRange.since, until: dateRange.until, generatedAt: new Date().toISOString(), scanRange: workspaceDirs, staleMaxAge },
    summary: { scannedRepoCount: repos.length, activeProjectCount: activeProjects.length, totalCommitCount, totalUncommittedFileCount, totalStaleFileCount },
    activeProjects, inactiveProjects, warnings: []
  }
}

function readStdin() {
  return new Promise(resolve => {
    let data = ''
    process.stdin.setEncoding('utf8')
    process.stdin.on('data', chunk => { data += chunk })
    process.stdin.on('end', () => { resolve(data) })
  })
}

function postJson(url, payload) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(payload)
    const target = new URL(url)
    const req = https.request({ hostname: target.hostname, path: target.pathname + target.search, method: 'POST', headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) } }, res => {
      let data = ''
      res.on('data', chunk => { data += chunk })
      res.on('end', () => { try { resolve(JSON.parse(data)) } catch (err) { resolve({ raw: data }) } })
    })
    req.on('error', reject)
    req.write(body)
    req.end()
  })
}

async function notifyWechatWork(config, content) {
  const wechatWork = config.wechatWork || {}
  if (!wechatWork.enabled) return { ok: false, error: { code: 'WECHAT_WORK_DISABLED', message: '企业微信机器人未启用。' } }
  if (!wechatWork.webhookUrl) return { ok: false, error: { code: 'WECHAT_WORK_WEBHOOK_REQUIRED', message: '未配置企业微信机器人 webhookUrl。' } }
  let finalContent = content
  if (wechatWork.mentionAll) finalContent += '\n\n<@all>'
  const result = await postJson(wechatWork.webhookUrl, { msgtype: 'markdown', markdown: { content: finalContent } })
  if (result.errcode && result.errcode !== 0) return { ok: false, error: { code: 'WECHAT_WORK_SEND_FAILED', message: result.errmsg || '企业微信机器人发送失败。' } }
  return { ok: true, result }
}

async function main() {
  const config = readConfig()
  const args = parseArgs(process.argv)
  if (args.command === 'collect') { outputJson(collect(config, args)); return }
  if (args.command === 'notify') { const content = await readStdin(); const result = await notifyWechatWork(config, content); outputJson(result); return }
  outputJson({ ok: false, error: { code: 'UNKNOWN_COMMAND', message: '未知命令。支持：collect、notify。' } })
}

main().catch(err => { outputJson({ ok: false, error: { code: 'RUNTIME_ERROR', message: err.message } }) })
