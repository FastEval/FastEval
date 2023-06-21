export function createBackToMainPageE() {
    const linkE = document.createElement('a')
    linkE.classList.add('back-to-main-page')
    linkE.textContent = '← Back to main page'
    linkE.href = '#'
    return linkE
}
