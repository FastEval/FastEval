import { createBenchmarksV } from './benchmarks/main.js'
import { parseHash } from './utils.js'

function toSorted(compareFn) {
    return [...this].sort(compareFn)
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

    const url = location.hostname === 'tju01.github.io' || branch !== 'main'
        ? 'https://raw.githubusercontent.com/tju01/ilm-eval/' + branch + '/reports'
        : './reports'

    document.body.textContent = 'Loading. May take a few seconds...'
    const benchmarksV = await createBenchmarksV(url)
    document.body.replaceChildren(benchmarksV)
}

main()
