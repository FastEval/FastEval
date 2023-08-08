import { createBenchmarksE } from './benchmarks/main.js'
import { computeUpdatedHash, parseHash } from './utils.js'

function toSorted(compareFn) {
    return [...this].sort(compareFn)
}

async function updateUrlIfBranchDoesntExistAnymore(branch) {
    const response = await fetch('https://raw.githubusercontent.com/fasteval/FastEval/' + branch + '/reports/__index__.json')
    if (!response.ok)
        location.hash = '#' + computeUpdatedHash({ branch: null })
}

async function main() {
    if (Array.prototype.toSorted === undefined)
        Array.prototype.toSorted = toSorted

    window.addEventListener('hashchange', () => {
        location.reload()
    })

    let branch = 'main'
    const hash = parseHash()
    if (hash.has('branch'))
        branch = hash.get('branch')

    if (branch !== 'main')
        updateUrlIfBranchDoesntExistAnymore(branch)

    const url = location.hostname === 'fasteval.github.io' || branch !== 'main'
        ? 'https://raw.githubusercontent.com/fasteval/FastEval/' + branch + '/reports'
        : './reports'

    document.body.textContent = 'Loading. May take a few seconds...'
    const benchmarksE = await createBenchmarksE(url)
    document.body.replaceChildren(benchmarksE)
}

main()
