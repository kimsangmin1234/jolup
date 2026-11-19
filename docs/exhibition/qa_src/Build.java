import kr.dogfoot.hwplib.object.HWPFile;
import kr.dogfoot.hwplib.object.bodytext.Section;
import kr.dogfoot.hwplib.object.bodytext.paragraph.Paragraph;
import kr.dogfoot.hwplib.object.bodytext.paragraph.header.ParaHeader;
import kr.dogfoot.hwplib.object.bodytext.paragraph.lineseg.LineSegItem;
import kr.dogfoot.hwplib.object.docinfo.*;
import kr.dogfoot.hwplib.tool.blankfilemaker.BlankFileMaker;
import kr.dogfoot.hwplib.writer.HWPWriter;
import java.nio.file.*;
import java.util.*;

// 사용: java Build lines.txt out.hwp
public class Build {
    static HWPFile f; static Section sec;
    static int cs(int size, boolean bold, int r, int g, int b) {
        DocInfo di = f.getDocInfo();
        CharShape c = di.getCharShapeList().get(0).clone();
        c.setBaseSize(size); c.getProperty().setBold(bold);
        c.getCharColor().setR((short) r); c.getCharColor().setG((short) g); c.getCharColor().setB((short) b);
        di.getCharShapeList().add(c);
        di.getIDMappings().setCharShapeCount(di.getCharShapeList().size());
        return di.getCharShapeList().size() - 1;
    }
    static int ps(int baseId, int left, int top, int bottom) {
        DocInfo di = f.getDocInfo();
        ParaShape p = di.getParaShapeList().get(baseId).clone();
        p.setLeftMargin(left); p.setIndent(0); p.setTopParaSpace(top); p.setBottomParaSpace(bottom);
        di.getParaShapeList().add(p);
        di.getIDMappings().setParaShapeCount(di.getParaShapeList().size());
        return di.getParaShapeList().size() - 1;
    }
    static void para(String text, int paraShape, int charShape, int height) throws Exception {
        Paragraph p = sec.addNewParagraph();
        ParaHeader h = p.getHeader();
        h.setLastInList(true);
        h.getControlMask().setValue(0);
        h.setParaShapeId(paraShape);
        h.setStyleId((short) 0);
        h.getDivideSort().setValue((short) 0);
        h.setCharShapeCount(1); h.setRangeTagCount(0); h.setLineAlignCount(1);
        h.setInstanceID(0); h.setIsMergedByTrack(0);
        p.createText(); p.getText().addString(text);
        h.setCharacterCount(text.length() + 1);
        p.createCharShape(); p.getCharShape().addParaCharShape(0, charShape);
        p.createLineSeg();
        LineSegItem s = p.getLineSeg().addNewLineSegItem();
        s.setTextStartPosition(0); s.setLineVerticalPosition(0);
        s.setLineHeight(height); s.setTextPartHeight(height);
        s.setDistanceBaseLineToLineVerticalPosition((int) (height * 0.85)); s.setLineSpace((int) (height * 0.6));
        s.setStartPositionFromColumn(0); s.setSegmentWidth(42520);
        s.getTag().setValue(393216);
    }
    public static void main(String[] a) throws Exception {
        f = BlankFileMaker.make();
        sec = f.getBodyText().getSectionList().get(0);
        int basePs = sec.getParagraph(0).getHeader().getParaShapeId();
        int csT = cs(1700, true, 0x14, 0x23, 0x3F), csM = cs(1000, false, 0x6B, 0x76, 0x86), csN = cs(1000, false, 0xA0, 0x4A, 0x00);
        int csS = cs(1400, true, 0x14, 0x23, 0x3F), csQ = cs(1100, true, 0x1F, 0x26, 0x33), csA = cs(1050, false, 0x1F, 0x26, 0x33);
        int psT = ps(basePs, 0, 0, 400), psS = ps(basePs, 0, 2000, 600), psQ = ps(basePs, 0, 900, 200), psA = ps(basePs, 1400, 0, 100);
        for (String line : Files.readAllLines(Paths.get(a[0]))) {
            if (line.isEmpty()) continue;
            String k = line.substring(0, 1), t = line.substring(2);
            switch (k) {
                case "T": para(t, psT, csT, 1700); break;
                case "M": para(t, psT, csM, 1000); break;
                case "N": para(t, psQ, csN, 1000); break;
                case "S": para(t, psS, csS, 1400); break;
                case "Q": para(t, psQ, csQ, 1100); break;
                default:  para(t, psA, csA, 1050);
            }
        }
        int n = sec.getParagraphCount();
        for (int i = 0; i < n; i++) sec.getParagraph(i).getHeader().setLastInList(i == n - 1);
        HWPWriter.toFile(f, a[1]);
        System.out.println("paragraphs=" + n);
    }
}
