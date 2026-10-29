"""Export initialized Geant4 geometry, check overlaps and draw actual dimensions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
KEEP = []


def draw_geometry(data, out):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetTitleFontSize(.04)
    colors = {name: ROOT.TColor.GetColor(color) for name, color in
              {'CsI': '#B9DDC2', 'PTFE': '#A7AFB9', 'Mylar': '#336CB0',
               'Grease': '#76C7CF', 'Sensor': '#CF735E', 'SourceExitFoil': '#777F89'}.items()}
    c = ROOT.TCanvas('geometry_inspection', 'Initialized Geant4 geometry', 1350, 740)
    def label(x, y, text, size=.032, ndc=True, color=1):
        t = ROOT.TLatex(x, y, text)
        t.SetTextFont(42); t.SetTextSize(size)
        t.SetTextColor(color)
        if ndc: t.SetNDC()
        else: t.SetTextAlign(22)
        t.Draw(); KEEP.append(t)
    def box(x1, y1, x2, y2, color, fill=True, width=1, style=1):
        b = ROOT.TBox(x1, y1, x2, y2)
        b.SetLineColor(color); b.SetLineWidth(width); b.SetLineStyle(style)
        b.SetFillColor(color); b.SetFillStyle(1001 if fill else 0)
        b.Draw(); KEEP.append(b)
    label(.035, .952, 'CsI(Tl) + Mylar + SiPM: geometry read from Geant4', .033)
    label(.035, .045, 'Confirmed: 10 mm CsI cube, 5 #mum front Mylar, PTFE wrap; no preamplifier.', .026)
    label(.035, .013, 'Assumed: source gap, coupling/PTFE thickness, 4 #times 4 array and all 16 channels summed.', .026)
    left = ROOT.TPad('xz', 'Longitudinal section', .015, .13, .495, .905)
    right = ROOT.TPad('xy', 'Readout mask', .505, .13, .985, .905)
    left.Draw(); right.Draw(); KEEP.extend([left, right])
    volumes = data['volumes']
    source = data['source']['center_mm']
    visible = [v for v in volumes if v['name'] != 'World']
    xhalf = max(8., max(abs(v['center_mm'][0])+v['size_mm'][0]/2 for v in visible)+1.5)
    zmin = min(-8., source[2]-2.)
    for pad in (left, right):
        pad.SetLeftMargin(.13); pad.SetRightMargin(.04)
        pad.SetBottomMargin(.13); pad.SetTopMargin(.08); pad.SetTicks(1, 1)
    left.cd()
    frame = left.DrawFrame(-xhalf, zmin, xhalf, 8, 'XZ section at y = 0;x (mm);z (mm)'); KEEP.append(frame)
    for v in volumes:
        if v['name'] == 'World': continue
        if v['rotation'] != [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]:
            raise ValueError('Plot requires axis-aligned boxes; JSON retains rotations')
        x, y, z = v['center_mm']; dx, dy, dz = [s/2 for s in v['size_mm']]
        if abs(y) > dy: continue
        box(x-dx, z-dz, x+dx, z+dz, colors[v['name']])
        if v['name'] == 'Mylar':
            line = ROOT.TLine(x-dx, z, x+dx, z)
            line.SetLineColor(colors['Mylar']); line.SetLineWidth(3); line.Draw(); KEEP.append(line)
    marker = ROOT.TMarker(source[0], source[2], 29)
    marker.SetMarkerColor(ROOT.kBlack); marker.SetMarkerSize(1.8); marker.Draw(); KEEP.append(marker)
    label(0., 0., 'CsI(Tl): 10 #times 10 #times 10 mm^{3}', .027, False)
    label(0., -7.1, 'Front Mylar: 5 #mum (blue line enhanced)', .025, False, colors['Mylar'])
    label(0., 6.4, 'SiPM entrance / coupling layer', .025, False)
    right.cd()
    frame = right.DrawFrame(-8, -8, 8, 8, 'XY readout mask viewed along z;x (mm);y (mm)'); KEEP.append(frame)
    sensor = next(v for v in volumes if v['name'] == 'Sensor')
    sx, sy, _ = sensor['size_mm']
    box(-sx/2, -sy/2, sx/2, sy/2, ROOT.kGray+1, False)
    mask = data['channel_mask']; nx = mask['array_nx']; ny = mask['array_ny']
    pitch = mask['pitch_mm']; active = mask['active_mm']; connected = set(mask['summed_channels'])
    for iy in range(ny):
        for ix in range(nx):
            i = iy*nx+ix; x = (ix-(nx-1)/2)*pitch; y = (iy-(ny-1)/2)*pitch
            box(x-active/2, y-active/2, x+active/2, y+active/2,
                colors['Sensor'] if i in connected else ROOT.kGray, i in connected)
            label(max(-4.4, min(4.4, x)), max(-4.4, min(4.4, y)), str(i), .027, False)
    crystal = next(v for v in volumes if v['name'] == 'CsI')
    dx, dy, _ = crystal['size_mm']
    box(-dx/2, -dy/2, dx/2, dy/2, ROOT.TColor.GetColor('#1A6B36'), False, 3, 2)
    label(.19, .865, 'Green outline: 10 mm crystal footprint', .028)
    label(.19, .175, '3 #times 3 mm^{2} active / 3.2 mm pitch', .028)
    c.cd()
    label(.59, .112, 'Scoring masks; not semiconductor volumes', .024)
    foil = data['config']['geometry'].get('source_exit_foil')
    source_note = f"Source gap: {data['config']['geometry']['source_gap_mm']:g} mm (assumed)"
    if foil: source_note += f"; exit foil: {foil['thickness_mm']:g} mm {foil['material']} (hypothesis)"
    label(.035, .079, source_note, .024)
    c.Modified(); c.Update()
    for extension in ('svg', 'png'): c.SaveAs(str(out/f'geometry.{extension}'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=HERE/'configs/csi_tl_initial.json')
    p.add_argument('--source', choices=['Am241', 'Co60'], default='Am241')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists(): p.error('Output directory must be new')
    build = subprocess.run(['cmake', '--build', str(HERE/'build'), '-j4'], capture_output=True, text=True)
    build.check_returncode()
    args.out.mkdir(parents=True); out = args.out.resolve()
    (out/'build.log').write_text(build.stdout+build.stderr)
    (out/'config.json').write_bytes(args.config.read_bytes())
    macro = out/'geometry.mac'
    macro.write_text(f'/csi/config {out}/config.json\n/csi/source {args.source}\n/csi/seed 271027\n/run/initialize\n'
                     '/geometry/test/resolution 10000\n/geometry/test/tolerance 0 mm\n'
                     '/geometry/test/run placed\n'
                     f'/csi/exportGeometry {out}/geometry.json\n'
                     '/vis/open VRML2FILE\n/vis/drawVolume\n'
                     '/vis/viewer/set/style surface\n/vis/viewer/set/viewpointThetaPhi 70 35 deg\n'
                     '/vis/viewer/flush\n')
    env = dict(os.environ, G4VRMLFILE_DEST_DIR=str(out)+'/',
               G4VRMLFILE_FILE_NAME='geometry.wrl', G4VRMLFILE_VIEWER='NONE')
    result = subprocess.run([str(HERE/'build/csi_sources'), str(macro)], env=env, capture_output=True, text=True)
    log = result.stdout+result.stderr; (out/'geometry.log').write_text(log)
    result.check_returncode()
    if 'Overlap is detected' in log or 'GeomVol1002' in log:
        raise RuntimeError('Geant4 reported an overlap; see geometry.log')
    if log.count('OK!') < 16: raise RuntimeError('Overlap check did not report all placements')
    data = json.loads((out/'geometry.json').read_text())
    if not (out/'geometry.wrl').is_file(): raise RuntimeError('VRML export missing')
    draw_geometry(data, out)
    report = {'passed': True, 'placements_including_world': len(data['volumes']),
              'overlap_check': 'Geant4 placed mode; 10000 sampled surface points per placement; zero tolerance',
              'scope': 'Initialized geometry and randomized overlap check; not verification of unknown apparatus parameters',
              'executable_sha256': hashlib.sha256((HERE/'build/csi_sources').read_bytes()).hexdigest(),
              'artifacts_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                   for f in (out/'geometry.json', out/'geometry.wrl', out/'config.json')}}
    (out/'inspection.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__': main()
